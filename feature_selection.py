from collections import namedtuple
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
from functools import cached_property
from functools import total_ordering
import io
import itertools
import logging
import os
from timeit import default_timer as timer
from typing import TypeVar

import humanize
import json
import optuna
from optuna.samplers import TPESampler
import pandas as pd
from scipy.special import comb
from trainkit import compute_loss
from trainkit import get_splits
from trainkit import split_by_counts

from sklearn.metrics import root_mean_squared_error


# optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("trainkit").setLevel(logging.INFO)
logger = logging.getLogger("feature_selection")
logger.setLevel(logging.INFO)


def feature_removal(
    model_params: dict[str, any],
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    test: tuple[pd.DataFrame, pd.Series] = tuple(),
    **kwargs,
):
    test_count = test[0].shape[0] if test else 0
    X = pd.concat([X_train, X_valid], axis=0)
    y = pd.concat([y_train, y_valid], axis=0)
    if test_count:
        X = pd.concat([X, test[0]], axis=0)
        y = pd.concat([y, test[1]], axis=0)
    return feature_removal_cv(
        model_params,
        X,
        y,
        split_count=split_by_counts(
            X_train.shape[0], X_valid.shape[0], test_count
        ),
        **kwargs,
    )


def feature_removal_cv(
    model_params: dict[str, any],
    X: pd.DataFrame,
    y: pd.Series,
    split_count,
    data_dir: str = os.getcwd(),
    trial_count: int = 100,
    max_removal_count: int = 5,
    always_keep: set[str] = set(),
    removal_suggestions: list[tuple[str]] = [],
    try_all_double_suggestions: bool = False,
):
    loss_state_path = f"{data_dir}/feature-removal-{y.name}-state.json"

    featurelist_path = f"{data_dir}/feature-removal-{y.name}-report.txt"
    if os.path.exists(featurelist_path):
        logger.warning(
            "Replacing feature removal report already saved to %s",
            featurelist_path,
        )

    if os.path.exists(loss_state_path):
        with open(loss_state_path, "r") as fp:
            objective = OptunaFeatureSelectionObjective.from_json(
                json.load(fp)
            )
        logger.info(
            "[feature_removal] Loaded %d losses from %s",
            len(objective.known_losses),
            loss_state_path,
        )
    else:
        logger.warning(
            "[feature_removal] (%s) Data size: %d", y.name, y.shape[0]
        )

        study = optuna.create_study(
            direction="minimize", sampler=TPESampler(seed=42)
        )

        # We first try the model using all features
        features = list(X.columns)

        study.enqueue_trial({ft: True for ft in features})
        logger.info(
            "[feature_removal] Original suggestions: %d",
            len(removal_suggestions),
        )
        # Suggest all single removals
        removal_suggestions += [(ft,) for ft in features]
        # Suggest all double removals
        if try_all_double_suggestions:
            removal_suggestions += itertools.combinations(features, 2)

        removal_suggestions = list(dict.fromkeys(removal_suggestions))
        for removals in removal_suggestions:
            suggestion = {ft: ft not in removals for ft in features}
            suggested_count = sum(suggestion.values())
            if suggested_count == 0 or suggested_count == len(features):
                logger.warning(
                    "[feature_removal] Suggestion count == %d; Skipped",
                    suggested_count,
                )
            else:
                logger.info(
                    "[feature_removal] Enqueue removal suggestion: %s",
                    [ft for ft in features if ft in removals],
                )
                study.enqueue_trial(suggestion)
        logger.warning(
            "[feature_removal] Final suggestions: %d",
            1 + len(removal_suggestions),
        )

        objective = OptunaFeatureSelectionObjective(
            model_params=model_params,
            X=X,
            y=y,
            splits=(
                get_splits(X, split_count)
                if isinstance(split_count, int)
                else split_count
            ),
            trial_count=trial_count,
            max_removal_count=max_removal_count,
            always_keep=always_keep,
        )
        with objective:
            study.optimize(
                objective,
                n_trials=trial_count,
            )
            with open(loss_state_path, "w") as fp:
                json.dump(objective.as_json, fp)

    objective.write_full_report(data_dir)
    logger.warning("[feature_removal] Final report written to %s", data_dir)


TrialInfoBase = namedtuple(
    "TrialInfo", "trial valid_trials improvement_percent loss removed_features"
)
TrialInfoBase.__new__.__defaults__ = (0, 0, 0, 0, [])


@total_ordering
class TrialInfo(TrialInfoBase):

    def __lt__(self, other):
        return self.loss > other.loss

    def __eq__(self, other):
        return self.loss == other.loss

    def __str__(self):
        """String representation.

        Returns
        -------
        str
            Format:
              #trial (percent valid trial):   percent improvement (loss) [removed count]
            Example:
              #0007 (valid #006 ( 85.71%)):   0.0008% (183.0998) [ 1 removed]
        """
        return (
            f"#{self.trial:04n} (valid #{self.valid_trials:03n} "
            f"({self.valid_trial_percent: 7.2f}%)): "
            f"{self.improvement_percent: 8.4f}% ({self.loss:.4f}) "
            f"[{self.removed_count: 2n} removed]"
        )

    @property
    def valid_trial_percent(self):
        if not self.trial:
            return 0
        return 100 * self.valid_trials / self.trial

    @property
    def removed_count(self):
        return len(self.removed_features)


T = TypeVar("T")


class OptunaFeatureSelectionObjective:
    """Provides an Optuna objective function to help you choose the best subset of features.

    Also provide an ETA, and generates a final report of the feature selection process,
    including the best and last improvement trials, and the overall feature selection statistics.
    This allows you to get a better understanding of the most and least important features.

    Attributes:
        model_params (dict): Parameters of the model to be optimized.
        X (pd.DataFrame): The input features dataframe.
        y (pd.Series): The target variable series.
        splits (list): A list of tuples containing train and validation indices.
        trial_count (int): The number of trials to perform.
        max_removal_count (int): The maximum number of features to consider removing in a trial.
        always_keep (set): Set of feature names not to be considered for removal.
        loss_fn (function): Used for evaluating model performance.
    """

    def __init__(
        self,
        model_params: dict[str, any],
        X: pd.DataFrame,
        y: pd.Series,
        splits: list[tuple] = None,
        trial_count: int = 1000,
        max_removal_count: int = 5,
        always_keep: set[str] = set(),
        loss_fn=root_mean_squared_error,
    ):
        self.model_params = model_params
        self.X = X
        self.y = y
        self.splits = splits
        self.trial_count = trial_count
        self.max_removal_count = max_removal_count
        self.always_keep: set[str] = always_keep
        self.loss_fn = loss_fn
        self.features = [] if X is None else list(X.columns)
        self.known_losses: dict[tuple[str], float] = {}
        self.optuna_repeated_count: int = 0
        self.original_loss: float = 0
        self.duration = 0
        self.hduration = None
        self.best_improvement_trial: TrialInfo = None
        self.last_improvement_trial: TrialInfo = None

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """Evaluates the loss of a model with a specific set of features.

        Called by the Optuna framework for each trial.


        Parameters
        ----------
        trial : optuna.trial.Trial

        Returns
        -------
        float
            Returns the loss which Optuna uses to determine the best set of features.

        """
        removed_features = self._early_stop(
            self._compute_removed_features(trial)
        )

        loss = self._compute_loss(removed_features)

        return self._handle_current_trial(trial, removed_features, loss).loss

    def __enter__(self):
        logger.info(
            f"[OptunaFeatureSelectionObjective] Target: {self.y.name}; "
            f"Size: {self.y.shape[0]}"
            f"\n  {len(list(self.X))} features: {self.features}"
            f"\n  {len(self.always_keep)} Always keep: {tuple(self.always_keep)}"
        )
        self.start_time = timer()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._update_stats()
        logger.info(
            "[OptunaFeatureSelectionObjective] %s", self.study_report()
        )

    def _early_stop(self, removed_features: T) -> T:
        if len(removed_features) > self.max_removal_count:
            logger.debug(
                "### TrialPruned: Removal count too high: %d",
                len(removed_features),
            )
            raise optuna.exceptions.TrialPruned()

        if len(removed_features) == len(self.features):
            logger.debug("### TrialPruned: No features left.")
            raise optuna.exceptions.TrialPruned()

        # Keep at least one item of always_keep:
        if self.always_keep and len(
            set(removed_features) & self.always_keep
        ) == len(self.always_keep):
            logger.debug("### TrialPruned: All required features were removed")
            raise optuna.exceptions.TrialPruned()

        if removed_features in self.known_losses:
            self.optuna_repeated_count += 1
            raise optuna.exceptions.TrialPruned()

        return removed_features

    def _compute_loss(self, removed_features: tuple[str]) -> float:
        selected_features = [
            feature_name
            for feature_name in self.features
            if feature_name not in removed_features
        ]

        loss = compute_loss(
            self.model_params,
            self.X,
            self.y,
            self.splits,
            selected_features,
            self.loss_fn,
        )
        self.known_losses[removed_features] = loss
        if len(removed_features) == 0:
            self.original_loss = loss
        return loss

    def _handle_current_trial(
        self, trial, removed_features: tuple[str], loss: float
    ) -> TrialInfo:
        improvement_percent = self.relative_improvement_percent(loss)
        current_trial = TrialInfo(
            trial.number + 1,
            self.combination_count,
            improvement_percent,
            loss,
            removed_features,
        )
        if improvement_percent >= 0:
            self.last_improvement_trial = current_trial
            if (
                self.best_improvement_trial is None
                or current_trial.loss < self.best_improvement_trial.loss
            ):
                self.best_improvement_trial = current_trial
        self.study_progress(current_trial)
        return current_trial

    def _update_stats(self):
        if not self.duration:
            self.duration = timer() - self.start_time
        if self.hduration is not None:
            return
        self.hduration = humanize.precisedelta(
            timedelta(seconds=self.duration)
        )
        self.valid_trial_mean_time = humanize.precisedelta(
            timedelta(
                seconds=(
                    (self.duration / self.combination_count)
                    if self.combination_count > 0
                    else 0
                )
            )
        )

    def eta(self, current_trial):
        duration = timer() - self.start_time
        remaining_trials = self.trial_count - current_trial
        result = remaining_trials * duration / current_trial
        return datetime.now() + timedelta(seconds=result)

    def relative_improvement_percent(self, loss: float) -> float:
        """Compares a loss against the baseline loss (with no features dropped).

        NEGATIVE values indicate an INCREASE in loss.
        POSITIVE values indicate  a DECREASE in loss.

        Parameters
        ----------
        loss : float
            Loss value for comparison.

        Returns
        -------
        float
            100 -> No loss at all (perfect prediction)
            50 -> 50% lower (half) the original loss
            0 -> Same loss
            -50 -> 50% greater loss than the original
            -100 -> 100% greater (twice) the original loss
        """
        if not self.original_loss:
            return 0
        return 100 * (1 - loss / self.original_loss)

    def _compute_removed_features(self, trial) -> tuple[str]:
        feature_selection = [
            trial.suggest_categorical(name, [True, False])
            for name in self.features
        ]
        return tuple(
            name
            for name, selected in zip(self.features, feature_selection)
            if not selected
        )

    @property
    def combination_count(self) -> int:
        return len(self.known_losses)

    @cached_property
    def positive_trial_percent(self) -> int:
        return (
            (
                100
                * (self.removal_rank["RI%"] > 0).sum()
                / len(self.known_losses)
            )
            if self.known_losses
            else 0
        )

    @cached_property
    def feature_stats(self) -> dict[str, int]:
        highest_percentile = defaultdict(lambda: float("-inf"))
        trial_counts = defaultdict(int)
        for _, rr in self.removal_rank.iterrows():
            for feature in rr.removed_features or [tuple()]:
                highest_percentile[feature] = max(
                    highest_percentile[feature], rr.percentile
                )
                trial_counts[feature] += 1

        return {
            f: {
                "highest_percentile": highest_percentile[f],
                "trial_count": trial_counts[f],
            }
            for f in self.features
        }

    @cached_property
    def removal_rank(self) -> pd.DataFrame:
        result = (
            (
                self.known_losses[k],
                k,
            )
            for k in sorted(
                self.known_losses.keys(),
                key=self.known_losses.get,
            )
        )
        df = pd.DataFrame(
            result,
            columns=["loss", "removed_features"],
        )
        df["RI%"] = df.loss.apply(self.relative_improvement_percent)
        df["percentile"] = df["RI%"].rank(pct=True) * 100
        df["rcount"] = df["removed_features"].apply(len)
        return df[["percentile", "RI%", "loss", "rcount", "removed_features"]]

    @cached_property
    def removal_rank_for_single_removals(self) -> pd.DataFrame:
        df = self.removal_rank
        df = df[df["rcount"] == 1].drop(columns=["rcount"])
        df["removed_features"] = df["removed_features"].apply(lambda f: f[0])
        df["tcount"] = df["removed_features"].apply(
            lambda f: self.feature_stats[f]["trial_count"]
        )
        df["highest_percentile"] = df["removed_features"].apply(
            lambda f: self.feature_stats[f]["highest_percentile"]
        )
        df = df.sort_values(
            by="highest_percentile", ascending=False
        ).reset_index(drop=True)
        return df[
            [
                "highest_percentile",
                "percentile",
                "removed_features",
                "RI%",
                "loss",
                "tcount",
            ]
        ]

    @cached_property
    def removal_rank_by_removal_count(self) -> pd.DataFrame:
        df = self.removal_rank
        df["rcount"] = df["removed_features"].apply(len)

        # Group by rcount and get the row with the minimum loss in each group
        grouped_df = df.loc[df.groupby("rcount")["loss"].idxmin()]
        group_counts = df["rcount"].value_counts().reset_index()
        group_counts.columns = ["rcount", "element_count"]
        grouped_df = grouped_df.merge(group_counts, on="rcount")

        sorted_df = grouped_df.sort_values(by="loss").reset_index(drop=True)
        sorted_df["improvement_rank"] = range(1, len(sorted_df) + 1)
        sorted_df["E%"] = (
            100
            * sorted_df["element_count"]
            / sorted_df["rcount"].apply(
                lambda v: comb(len(self.features), v, exact=True)
            )
        )

        # Sort by rcount in reverse order
        sorted_df = sorted_df.sort_values(by="rcount", ascending=False)
        return sorted_df[
            [
                "percentile",
                "improvement_rank",
                "RI%",
                "loss",
                "element_count",
                "E%",
                "rcount",
                "removed_features",
            ]
        ]

    def study_progress(self, current_trial):
        if not current_trial:
            clear_line(9)
        show_best = (
            self.best_improvement_trial is not None
            and self.best_improvement_trial.trial != current_trial.trial
        )
        show_last = (
            self.best_improvement_trial is None
            or self.best_improvement_trial.trial
            != self.last_improvement_trial.trial
        ) and current_trial.trial != self.last_improvement_trial.trial
        show_best_str = (
            f"\n     Best: {self.best_improvement_trial}"
            if show_best
            else "\n     Best:"
        )
        show_best_removal_str = (
            f"\n     Best: {self.best_improvement_trial.removed_features}"
            if show_best or True
            else "\n     Best:"
        )
        show_last_str = (
            f"\n    Last+: {self.last_improvement_trial}"
            if show_last
            else "\n    Last+:"
        )
        show_last_removal_str = (
            f"\n    Last+: {self.last_improvement_trial.removed_features}"
            if show_last
            else "\n    Last+:"
        )
        show_last_removal_str = ""
        logger.info(  # TODO print
            f"\n## [ETA {self.eta(current_trial.trial)}] Valid trials: "
            f"{100 * self.combination_count / (current_trial.trial):02.2f}% "
            f"({self.combination_count})"
            f"\n   Trials:"
            f"{show_best_str}{show_last_str}"
            f"\n     Cur.: {current_trial}"
            f"\n   Removals:"
            f"{show_best_removal_str}{show_last_removal_str}"
            # f"\n     Cur.: {current_trial.removed_features}"
        )

    def study_report(self):
        self._update_stats()
        report = (
            f"\nTarget        : {self.y.name}; Rows: {self.y.shape[0]};"
            f" Test data: {len(self.splits[0]) > 2 if self.splits is not None else '?'}"
            f"\nTime          : {self.hduration}"
            f"\nMax removal   : {self.max_removal_count}"
            f"\nFeatures      : {len(self.features)}"
            f"\n  {self.features}"
            f"\nAlways keep   : {len(self.always_keep)}"
            f"\n  {tuple(self.always_keep)}"
            f"\nTrials (Total): {self.trial_count}"
            f"\n  Repeated    : {100 * self.optuna_repeated_count / self.trial_count:02.1f}%"
            f"\nTrials (Valid): {self.combination_count}"
            f"\n  % of total  : {100 * self.combination_count / self.trial_count:02.2f}%"
            f"\n  % Positive  : {self.positive_trial_percent:02.2f}%"
            f"\n  Mean time   : {self.valid_trial_mean_time}"
            f"\nImprovement   :"
            f"\n  Best: {self.best_improvement_trial}"
            f"\n  Last: {self.last_improvement_trial}"
            f"\nRemovals:"
            f"\n  Best: {self.best_improvement_trial.removed_features if self.best_improvement_trial else None}"
        )

        report += (
            "\n\n---\nRemoval count ranking (showing best entry for each removal count)"
            "\n  - Best is always at 'improvement-rank'=1"
            "\n\nPercentile, improvement-rank, relative-improvement-%, loss, element-count, E%, removed-count, removed\n"
        )
        report += self.print_removal_count_ranking_rows()

        with io.StringIO() as sio:
            sio.write(
                "\n\n---\nRelative improvement percent (RI%) ranking for single removals, from best to worst:"
                "\n  - Relative to the baseline (no features removed)"
                "\n\nhighest-percentile, percentile, RI%, loss, trial-count, removed\n"
            )
            self.write_relative_loss_ranking_rows_for_single_removal(sio)

            sio.write(
                "\n\n---\nRelative improvement percent (RI%) ranking, from best to worst:"
                "\n  - Relative to the baseline (no features removed)"
                "\n\nPercentile, RI%, loss, removed-count, removed\n"
            )
            self.write_relative_loss_ranking_rows(sio)
            return report + sio.getvalue()

    def print_removal_count_ranking_rows(self) -> str:
        result = ""
        for _, row in self.removal_rank_by_removal_count.iterrows():
            result += (
                f"{row.percentile:05.1f}, "
                f"{row.improvement_rank:02n}, "
                f"{row['RI%']: 08.4f}, "
                f"{row.loss:5.1f}, "
                f"{row.element_count:03n}, "
                f"{row['E%']:05.1f}, "
                f"{row.rcount:02n}, "
                f"{row.removed_features}\n"
            )

        return result

    def write_relative_loss_ranking_rows_for_single_removal(self, fp):
        self.write_relative_loss_ranking_rows(
            fp, self.removal_rank_for_single_removals
        )

    def write_relative_loss_ranking_rows(self, fp, df: pd.DataFrame = None):
        if df is None:
            df = self.removal_rank

        for _, row in df.iterrows():
            if "highest_percentile" in df.columns:
                fp.write(f"{row.highest_percentile: 6.1f}, ")
            if "percentile" in df.columns:
                fp.write(f"{row.percentile: 6.1f}, ")
            fp.write(f"{row['RI%']: 8.4f}, " f"{row.loss: 6.1f}, ")
            if "rcount" in df.columns:
                fp.write(f"{row.rcount: 2n}, ")
            if "tcount" in df.columns:
                fp.write(f"{row.tcount: 4n}, ")
            fp.write(f"{row.removed_features}\n")

    def write_full_report(self, data_dir):
        report_path = f"{data_dir}/feature-removal-{self.y.name}-report.txt"
        self.removal_rank_by_removal_count.to_csv(
            f"{data_dir}/feature-removal-{self.y.name}-by-removal-counts.csv"
        )
        self.removal_rank_for_single_removals.to_csv(
            f"{data_dir}/feature-removal-{self.y.name}-single-removals.csv"
        )
        self.removal_rank.to_csv(
            f"{data_dir}/feature-removal-{self.y.name}-by-relative-improvement.csv"
        )

        with open(report_path, "w") as optuna_fp:
            optuna_fp.write(self.study_report())

    @property
    def as_json(self) -> dict[str, any]:
        self._update_stats()
        return {
            "target": self.y.name,
            "size": self.y.shape[0],
            "features": self.features,
            "always_keep": list(self.always_keep),
            "max_removal_count": self.max_removal_count,
            "trial_count": self.trial_count,
            "optuna_repeated_count": self.optuna_repeated_count,
            "start_time": self.start_time,
            "duration": self.duration,
            "best_improvement_trial": self.best_improvement_trial,
            "known_losses": list(self.known_losses.items()),
        }

    @classmethod
    def from_json(cls, json_dict: dict):
        sparse_index = pd.RangeIndex(
            start=0, stop=json_dict.get("size", 0), step=1
        )
        result = cls(
            None,
            None,
            y=pd.Series(
                name=json_dict.get("target", "no-name"),
                index=sparse_index,
            ),
            trial_count=json_dict.get("trial_count", 1),
            max_removal_count=json_dict.get("max_removal_count", 0),
            always_keep=set(json_dict.get("always_keep", [])),
        )
        result.known_losses = {
            tuple(removed): loss for removed, loss in json_dict["known_losses"]
        }

        result.features = json_dict.get("features", [])
        result.optuna_repeated_count = json_dict.get(
            "optuna_repeated_count", 0
        )
        result.start_time = json_dict.get("start_time")
        result.duration = json_dict.get("duration")
        result.original_loss = result.known_losses.get(tuple(), -1)
        result.best_improvement_trial = TrialInfo(
            *(json_dict.get("best_improvement_trial", []) or [])
        )
        return result


def clear_line(n=1):
    LINE_UP = "\033[1A"
    LINE_CLEAR = "\x1b[2K"
    return  # TODO
    for i in range(n):
        print(LINE_UP, end=LINE_CLEAR)
