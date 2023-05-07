## Introduction

This is my "streamlined" version of [`socceraction`](https://github.com/ML-KULeuven/socceraction)'s notebooks for data processing and modeling.

Compared to the notebooks, the big changes are as follows:

1.  Parquet files are used instead of hdf5 for data storage.
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

There is also a script `R/3-evaluate-models` that can be used to calculate ROC AUC, log loss, and Brier score for the xgboost models. It's not essential to the workflow, but it's useful for comparing one's own models to those in the [VAEP paper](https://arxiv.org/pdf/1802.07127.pdf).