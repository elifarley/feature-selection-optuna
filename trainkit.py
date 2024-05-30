import logging
from typing import Callable
import warnings

import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold


logger = logging.getLogger("trainkit")


def split_by_counts(train_count: int, valid_count: int, test_count: int = 0):
    """Generate indices for training and validation sets based on specified counts.

    This function creates a list of tuples, where each tuple contains
    two numpy arrays: the first array contains indices for the training set,
    and the second array contains indices for the validation set.

    Parameters
    ----------
    train_count: int
        The row count of the training set.
    valid_count: int
        The row count of the validation set.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        A list containing one tuple of two numpy arrays, where the first array
        is the training indices and the second array is the validation indices.
    """
    result = (
        np.arange(train_count),
        np.arange(train_count, train_count + valid_count),
    )
    if test_count:
        result += (
            np.arange(
                train_count + valid_count,
                train_count + valid_count + test_count,
            ),
        )
    return [result]


def get_splits(X, split_count=2):
    """Generate training and validation indices for K-Fold cross-validation.

    A list of tuples is created in which each tuple contains two arrays:
    the first array contains indices for the training set, and
    the second array contains indices for the validation set.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        The data used for creating splits.
    split_count : int, optional
        The number of splits/folds to create. Default is 2.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of tuples containing the training indices and validation indices.
    """
    kf = KFold(n_splits=split_count, shuffle=True, random_state=42)
    return [(train_idx, valid_idx) for train_idx, valid_idx in kf.split(X)]


def compute_loss(
    model_params: dict[str, any],
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple],
    selected_feature_names: list[str] | None = None,
    loss_fn: Callable[[pd.Series, pd.Series], float] = root_mean_squared_error,
) -> float:
    """Compute the loss for a given model configuration and dataset.

    Parameters
    ----------
    model_params : dict[str, any]
        Parameters for the model to be trained.
    X : pd.DataFrame
        The input features dataframe containing corresponding values
        for both training and validation sets.
    y : pd.Series
        The target variable series containing corresponding values
        for both training and validation sets.
    splits : list[tuple]
        List of tuples of indices for training and validation sets,
        used to split X and y into respective training and validation subsets.
    selected_feature_names : list[str], optional
        List of feature names to be used. If None, all features in X are used.
    loss_fn : function, optional
        The loss function to be used. Defaults to root_mean_squared_error.

    Returns
    -------
    float
        The average loss computed over all the validation splits.
    """
    loss = 0
    iteration = 0
    if not selected_feature_names:
        selected_feature_names = list(X.columns)

    for split_item in splits:
        iteration += 1
        train_idx, valid_idx = (split_item[0], split_item[1])
        X_train_split = X.iloc[train_idx][selected_feature_names]
        y_train_split = y.iloc[train_idx]
        X_valid_split = X.iloc[valid_idx][selected_feature_names]
        y_valid_split = y.iloc[valid_idx]

        train_data = lgb.Dataset(X_train_split, label=y_train_split)
        valid_data = lgb.Dataset(
            X_valid_split, label=y_valid_split, reference=train_data
        )
        with warnings.catch_warnings():  # Python 3.11: (action="ignore"):
            warnings.simplefilter("ignore")
            gbm = lgb.train(
                model_params,
                train_data,
                valid_sets=[valid_data],
            )

        if len(split_item) == 3:
            X_test_split = X.iloc[split_item[2]][selected_feature_names]
            y_test_split = y.iloc[split_item[2]]
        else:
            X_test_split = X_valid_split
            y_test_split = y_valid_split

        pred = gbm.predict(X_test_split, num_iteration=gbm.best_iteration)
        iter_loss = loss_fn(y_test_split, pred)

        loss += iter_loss
        if iteration > 1:
            logger.debug(
                "[compute_loss] RMSE: %07.3f ( ~%07.3f)",
                iter_loss,
                loss / iteration,
            )

    return loss / len(splits)
