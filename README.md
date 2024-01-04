## Introduction

This is my "streamlined" version of [`socceraction`](https://github.com/ML-KULeuven/socceraction)'s notebooks for data processing and modeling.

Compared to the notebooks, the big changes are as follows:

1.  Parquet files are used instead of HDF5 for data storage.
2.  Modeling is done with R.

## Scripts

Scripts are meant to be run in the following order.

1.  `src/0-get-vaep-data.py`
    -   This assumes you have Opta event data in the folder specified in `RAW_DATA_DIR`, nested and named in the format specified by `feeds` in `socceraction.spadl.opta.OptaLoader()`.
    -   Processed data is sent to `PROCESSED_DATA_DIR`, nested in folders for `competition_id` and `season_id`
    -   The script is written to process just one competition and one season at a time. You would need to update `COMPETITION_ID` and `SEASON_ID` and re-run the script to process another competition and season.
2.  `R/1-combine-data.R`:
    -   Combine multiple competitions and seasons of data nested under `PROCESSED_DATA_DIR` into one new folder `FINAL_DATA_DIR` to prepare for modeling and so forth.
3.  `R/2-model-and-predict.R`
    -   Fit xgboost models for scoring and conceding probabilities.
4.  `R/3-convert-preds-to-vaep.R`
    -   Converts xgboost predictions to VAEP value (i.e. what `soccerction.vaep.formula.vaepformula()` and `soccerction.atomic.vaep.formula.vaepformula()` do)
5.  `R/4-post-process-vaep.R`
    -   Combine the VAEP values with the actions data set.
    -   Summarize pleyer seasons.

### Supplementary

-   `R/3-evaluate-models.R`
    -   This can be used to calculate ROC AUC, log loss, and Brier score for the xgboost models. It's not essential to the workflow, but it's useful for comparing one's own models to those in the [VAEP paper](https://arxiv.org/pdf/1802.07127.pdf).
-   `R/upload-to-github.R`
    -   I've used this to one-off upload data from local directories to GitHub releases. See the "Data" section for more info.
-   `R/helpers.R`
    -   Functions for other scripts.

## Data

### `data-processed` Release

The `data-processed` release contains data stored in the `PROCESSED_DATA_DIR` on local. (See output of `src/0-get-vaep-data.py`.) Note that data files in the repository itself are ignored due to GitHub's file hosting size limitations.)

The data is named in this format: `{league_id}-{season_id}-{concept}.parquet`

-   `league_id`: `8` for EPL, `85` for MLS
-   `season_id`: End season for EPL (e.g. `2023` = "2022/23 season"), season for MLS
-   `concept`: One of `actions`, `actiontypes`, `bodyparts`, `games`, `gamestate_actions`, `players`, `teams`, `x`, `y`, and `xt`.

More on `concept`:

-   The `concepts` correspond with the identically named tables in `socceraction`'s HDF5 format.
-   There are also "atomic" flavors of each--with the exception of `xt`--that are suffixed with `_atomic`. (`bodyparts`, `games`, `players`, and `teams` are not different for atomic data.)
-   `x` and `y` include the lagged features of the prior 2 actions.

Note that, on local, files are nested in sub-directories, e.g. `{PROCESSED_DATA_DIR}/{league_id}/{season_id}/{concept}.parquet`. Because GitHub releases don't have something equivalent to sub-directories, string concatenation is used.

### Models

Model files that are the output of `R/2-model-and-predict.R` are stored in the GitHub repo itself at `{FINAL_DATA_DIR}/{league_id}/{first_season_id}-{last_season_id}/`, where `{first_season_id}` and `{last_season_id}` represent the range of seasons used to train the model. For VAEP, there are two models `model_scores.model` and `model_concedes.model`. See the [VAEP paper](https://arxiv.org/pdf/1802.07127.pdf) for an explanation of the need for two models.
