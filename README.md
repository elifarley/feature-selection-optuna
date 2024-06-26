## Introduction

Feature selection is a critical step in machine learning to enhance model performance and reduce overfitting. However, evaluating every possible combination of features can be computationally expensive.

This library introduces an efficient approach to feature selection using Optuna, which significantly reduces the search space and computation time.

## Details

The class `OptunaFeatureSelectionObjective` conducts a study where each trial
attempts a different subset of features from the input dataset.

Key highlights:

### Pruning Strategy

Trials are pruned using `optuna.exceptions.TrialPruned` in three scenarios:

1. The set of removed features has been evaluated in a prevous trial.
2. The number of removed features exceeds a user-defined maximum.
3. No features are selected for the trial.

### ETA Prediction

After each successful trial, an ETA is printed, letting users know when the study is likely to complete.
This feature is particularly useful for long-running studies, allowing users to manage their time more efficiently.

## Example Usage

```Python
from feature_selection import feature_removal_cv

feature_removal_cv(
    model_params={
        "objective": "regression",
        "metric": "rmse",
        "data_random_seed": 42,
        "num_boost_round": 1000,
        "early_stopping_rounds": 10,
        "learning_rate": 0.12599281729053988,
        "force_row_wise": True,
        "verbose": -1,
        "verbose_eval": False,
        "num_leaves": 631,
        "max_depth": 7,
        "min_child_samples": 65,
        "colsample_bytree": 0.8430078242019065,
        "reg_alpha": 0.06636017620531826,
        "reg_lambda": 0.057077523364489346,
    },
    X=df.drop(columns=["MedHouseVal"]),
    y=df.MedHouseVal,
    split_count=5,
    trial_count=800,
)

```

## Example Report

Example feature removal report from the `fetch_california_housing` dataset,
using *LightGBM* as the model:

```
Target     : MedHouseVal; Rows: 20640
Time       : 4 minutes and 9.54 seconds
Max removal: 5
Trials     : 1000
  Repeated : 87.1%
  Valid    : 11.10% (111)
  Mean time: 2.25 seconds
Improvement:
  Best: #0012 (valid #12 (92.31%)):   0.5640% (0.4470) [02 removed]
  Last: #0012 (valid #12 (92.31%)):   0.5640% (0.4470) [02 removed]

---
Removal count ranking (showing best entry for each removal count)
  - Best is always at 'improvement-rank'=1

removed-count, improvement-rank, element-count, relative-improvement-%, loss, removed
05, 06, 016, -04.66230,   0.47045773007001, ('HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup')
04, 05, 030, -04.21248,   0.46843579420832, ('HouseAge', 'AveBedrms', 'Population', 'AveOccup')
03, 04, 032, -01.03054,   0.45413293903588, ('HouseAge', 'AveBedrms', 'Population')
02, 01, 024,  00.56401,   0.44696545144867, ('AveBedrms', 'Population')
01, 02, 008,  00.16590,   0.44875497579740, ('Population',)
00, 03, 001,  00.00000,   0.44950067535113, ()


---
Relative improvement percent (RI%) ranking for single removals, from best to worst:
  - Relative to the baseline (no features removed)

RI%, loss, removed-count, removed
 00.16590,   0.44875497579740, ('Population',)
 00.10078,   0.44904767131884, ('AveBedrms',)
-02.53074,   0.46087636019696, ('AveRooms',)
-02.67508,   0.46152517577751, ('HouseAge',)
-03.78445,   0.46651178215665, ('MedInc',)
-04.78708,   0.47101864931400, ('AveOccup',)
-25.20672,   0.56280506141426, ('Latitude',)
-26.80935,   0.57000889293308, ('Longitude',)


---
Relative improvement percent (RI%) ranking, from best to worst:
  - Relative to the baseline (no features removed)
\RI%, loss, removed-count, removed
 00.56401,   0.44696545144867, 02, ('AveBedrms', 'Population')
 00.16590,   0.44875497579740, 01, ('Population',)
 00.10078,   0.44904767131884, 01, ('AveBedrms',)
 00.00000,   0.44950067535113, 00, ()
-01.03054,   0.45413293903588, 03, ('HouseAge', 'AveBedrms', 'Population')
-02.08879,   0.45888982179241, 02, ('HouseAge', 'AveBedrms')
-02.19110,   0.45934969857630, 02, ('MedInc', 'Population')
-02.39523,   0.46026727013703, 03, ('AveRooms', 'AveBedrms', 'Population')
-02.42233,   0.46038904516264, 02, ('HouseAge', 'Population')
-02.53074,   0.46087636019696, 01, ('AveRooms',)
-02.60086,   0.46119154480805, 02, ('AveRooms', 'Population')
-02.67508,   0.46152517577751, 01, ('HouseAge',)
-03.12164,   0.46353248336082, 02, ('AveRooms', 'AveBedrms')
-03.78445,   0.46651178215665, 01, ('MedInc',)
```

<details>
<summary>
More rows...

You can see that the *single-column removal* that caused the worst loss (`0.57`) was `Longitude`:

```
-25.20672,   0.56280506141426, 1, ('Latitude',)
-25.70993,   0.56506699440362, 2, ('Population', 'Latitude')
-26.65316,   0.56930679300722, 5, ('MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population')
-26.80935,   0.57000889293308, 1, ('Longitude',)
```

</summary>

```csv
-3.96840,   0.46733864199451, 3, ('AveBedrms', 'Population', 'AveOccup')
-4.21248,   0.46843579420832, 4, ('HouseAge', 'AveBedrms', 'Population', 'AveOccup')
-4.21519,   0.46844796589609, 4, ('AveRooms', 'AveBedrms', 'Population', 'AveOccup')
-4.43502,   0.46943610903865, 2, ('Population', 'AveOccup')
-4.46202,   0.46955749077910, 4, ('HouseAge', 'AveRooms', 'AveBedrms', 'Population')
-4.66230,   0.47045773007001, 5, ('HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup')
-4.78708,   0.47101864931400, 1, ('AveOccup',)
-4.91582,   0.47159730635057, 2, ('AveBedrms', 'AveOccup')
-5.10043,   0.47242714725383, 3, ('MedInc', 'HouseAge', 'Population')
-5.14044,   0.47260699804074, 3, ('HouseAge', 'AveRooms', 'Population')
-5.21229,   0.47292996543346, 2, ('MedInc', 'HouseAge')
-5.38551,   0.47370856987459, 3, ('HouseAge', 'AveRooms', 'AveBedrms')
-5.41085,   0.47382246444499, 2, ('MedInc', 'AveBedrms')
-5.49691,   0.47420932364899, 3, ('MedInc', 'AveBedrms', 'Population')
-5.55052,   0.47445029715723, 4, ('HouseAge', 'AveRooms', 'AveBedrms', 'AveOccup')
-5.60144,   0.47467918116092, 2, ('HouseAge', 'AveRooms')
-5.61758,   0.47475172928213, 3, ('AveRooms', 'Population', 'AveOccup')
-5.73287,   0.47526995084483, 4, ('HouseAge', 'AveRooms', 'Population', 'AveOccup')
-6.32997,   0.47795392135789, 2, ('HouseAge', 'AveOccup')
-6.34165,   0.47800642784814, 3, ('HouseAge', 'Population', 'AveOccup')
-6.41758,   0.47834772942566, 2, ('AveRooms', 'AveOccup')
-7.08493,   0.48134748815935, 3, ('AveRooms', 'AveBedrms', 'AveOccup')
-7.27891,   0.48221941432214, 4, ('MedInc', 'HouseAge', 'AveBedrms', 'Population')
-7.70316,   0.48412642281568, 3, ('MedInc', 'HouseAge', 'AveBedrms')
-8.11176,   0.48596311332694, 3, ('MedInc', 'Population', 'AveOccup')
-8.14567,   0.48611550225646, 4, ('MedInc', 'HouseAge', 'Population', 'AveOccup')
-8.37244,   0.48713483231990, 3, ('HouseAge', 'AveRooms', 'AveOccup')
-8.63764,   0.48832693153420, 2, ('MedInc', 'AveOccup')
-9.05826,   0.49021761123812, 4, ('MedInc', 'AveBedrms', 'Population', 'AveOccup')
-9.10097,   0.49040961473868, 5, ('MedInc', 'HouseAge', 'AveBedrms', 'Population', 'AveOccup')
-10.45483,   0.49649522924872, 3, ('MedInc', 'AveBedrms', 'AveOccup')
-10.81821,   0.49812858168702, 3, ('MedInc', 'HouseAge', 'AveOccup')
-25.20672,   0.56280506141426, 1, ('Latitude',)
-25.70993,   0.56506699440362, 2, ('Population', 'Latitude')
-26.65316,   0.56930679300722, 5, ('MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population')
-26.80935,   0.57000889293308, 1, ('Longitude',)
-26.84404,   0.57016482383186, 3, ('AveBedrms', 'Population', 'Latitude')
-26.96979,   0.57073006845217, 2, ('AveBedrms', 'Longitude')
-27.07158,   0.57118760084460, 2, ('Population', 'Longitude')
-27.29190,   0.57217796406823, 2, ('AveRooms', 'Latitude')
-27.30702,   0.57224591383528, 4, ('MedInc', 'AveRooms', 'AveBedrms', 'Population')
-28.08064,   0.57572336260849, 3, ('AveBedrms', 'Population', 'Longitude')
-28.61308,   0.57811666308142, 3, ('MedInc', 'AveRooms', 'Population')
-29.03914,   0.58003180973113, 2, ('AveRooms', 'Longitude')
-29.41612,   0.58172635431420, 3, ('MedInc', 'AveRooms', 'AveBedrms')
-29.59896,   0.58254821793080, 4, ('AveRooms', 'AveBedrms', 'Population', 'Latitude')
-29.95871,   0.58416527803826, 3, ('AveRooms', 'AveBedrms', 'Longitude')
-30.78791,   0.58789252613351, 2, ('MedInc', 'AveRooms')
-32.07312,   0.59366956094864, 4, ('MedInc', 'HouseAge', 'AveRooms', 'Population')
-32.66674,   0.59633788722049, 4, ('HouseAge', 'AveBedrms', 'Population', 'Latitude')
-32.75321,   0.59672656085596, 5, ('MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'AveOccup')
-32.86823,   0.59724360912042, 4, ('MedInc', 'HouseAge', 'AveRooms', 'AveBedrms')
-33.73193,   0.60112592645904, 3, ('MedInc', 'HouseAge', 'AveRooms')
-33.74887,   0.60120205882995, 4, ('MedInc', 'AveRooms', 'AveBedrms', 'AveOccup')
-34.97764,   0.60672539730386, 4, ('HouseAge', 'AveRooms', 'Population', 'Latitude')
-35.41009,   0.60866927817798, 4, ('HouseAge', 'AveBedrms', 'Population', 'Longitude')
-35.66525,   0.60981622219018, 5, ('HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'Latitude')
-36.76957,   0.61478012261011, 4, ('MedInc', 'HouseAge', 'AveRooms', 'AveOccup')
-36.80242,   0.61492780263808, 3, ('HouseAge', 'AveRooms', 'Longitude')
-37.80189,   0.61942040462466, 4, ('HouseAge', 'AveRooms', 'AveBedrms', 'Longitude')
-37.91647,   0.61993544414150, 4, ('HouseAge', 'AveRooms', 'Population', 'Longitude')
-38.47016,   0.62242429219966, 5, ('HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'Longitude')
-39.59685,   0.62748878981688, 2, ('AveOccup', 'Latitude')
-40.65759,   0.63225683442081, 2, ('Latitude', 'Longitude')
-42.37149,   0.63996082283022, 3, ('Population', 'Latitude', 'Longitude')
-43.18696,   0.64362636179336, 2, ('AveOccup', 'Longitude')
-43.22096,   0.64377920005569, 3, ('AveBedrms', 'Latitude', 'Longitude')
-43.80332,   0.64639691374347, 4, ('AveBedrms', 'Population', 'AveOccup', 'Latitude')
-43.87200,   0.64670560924455, 3, ('AveBedrms', 'AveOccup', 'Latitude')
-44.34598,   0.64883615376219, 3, ('Population', 'AveOccup', 'Longitude')
-45.44961,   0.65379697705133, 3, ('HouseAge', 'AveOccup', 'Latitude')
-46.05747,   0.65652932544397, 4, ('AveBedrms', 'Population', 'Latitude', 'Longitude')
-46.94957,   0.66053930119182, 3, ('AveBedrms', 'AveOccup', 'Longitude')
-47.07703,   0.66111225482940, 4, ('HouseAge', 'Population', 'AveOccup', 'Latitude')
-47.74738,   0.66412547514768, 4, ('AveBedrms', 'Population', 'AveOccup', 'Longitude')
-49.16841,   0.67051300238039, 3, ('MedInc', 'Population', 'Latitude')
-49.69176,   0.67286549232064, 4, ('AveRooms', 'Population', 'AveOccup', 'Longitude')
-49.79569,   0.67333265521987, 4, ('AveRooms', 'AveBedrms', 'Latitude', 'Longitude')
-50.02319,   0.67435526750837, 5, ('HouseAge', 'AveBedrms', 'Population', 'AveOccup', 'Latitude')
-50.77163,   0.67771948267529, 3, ('HouseAge', 'AveOccup', 'Longitude')
-50.84316,   0.67804101425540, 2, ('MedInc', 'Longitude')
-51.96104,   0.68306592313965, 3, ('HouseAge', 'Latitude', 'Longitude')
-52.67413,   0.68627126128431, 5, ('HouseAge', 'AveRooms', 'AveBedrms', 'AveOccup', 'Latitude')
-52.83002,   0.68697195897298, 4, ('HouseAge', 'Population', 'AveOccup', 'Longitude')
-52.83559,   0.68699699230203, 5, ('AveRooms', 'AveBedrms', 'Population', 'Latitude', 'Longitude')
-54.87442,   0.69616156721228, 5, ('HouseAge', 'AveBedrms', 'Population', 'Latitude', 'Longitude')
-55.11743,   0.69725388310838, 4, ('MedInc', 'HouseAge', 'Population', 'Latitude')
-56.04720,   0.70143320037677, 4, ('MedInc', 'AveBedrms', 'Population', 'Latitude')
-61.36219,   0.72532414110002, 5, ('HouseAge', 'AveRooms', 'Population', 'Latitude', 'Longitude')
-62.85236,   0.73202247858911, 4, ('AveBedrms', 'AveOccup', 'Latitude', 'Longitude')
-75.93869,   0.79084560195901, 5, ('MedInc', 'AveBedrms', 'Population', 'AveOccup', 'Latitude')
-80.21695,   0.81007640807960, 3, ('MedInc', 'AveOccup', 'Longitude')
-87.01191,   0.84061982032198, 5, ('MedInc', 'AveBedrms', 'Population', 'AveOccup', 'Longitude')
-95.02615,   0.87664387066150, 4, ('MedInc', 'Population', 'Latitude', 'Longitude')
-99.66718,   0.89750532825562, 5, ('MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Latitude')
-101.97507,   0.90787931437800, 5, ('MedInc', 'HouseAge', 'Population', 'Latitude', 'Longitude')
-116.35262,   0.97250648930469, 5, ('MedInc', 'Population', 'AveOccup', 'Latitude', 'Longitude')
```

</details>
