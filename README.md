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
    -   Fit xgboost models for scoring and conceding probabilities, run predictions to generate `ovaep` and `dvaep`, and combine the predictions into `vaep`.
    -   Adds possession IDs to predictions.
4.  `R/3-post-process.R`
    -   Combine the VAEP predictions with the actions data set.
    -   Summarize pleyer seasons.
