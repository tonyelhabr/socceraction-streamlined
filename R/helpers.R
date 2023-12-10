library(arrow)
library(fs)

COMPETITION_ID <- 85
SEASON_END_YEARS <- c(2013:2019, 2021:2022)
TEST_SEASON_IDS <- c(2016:2019, 2021:2022) # SEASON_END_YEARS # max(SEASON_END_YEARS)
PROCESSED_DATA_DIR <- 'data/processed'
PARENT_FINAL_DATA_DIR <- 'data/final'
FINAL_DATA_DIR <- file.path(PARENT_FINAL_DATA_DIR, COMPETITION_ID, sprintf('%s-%s', min(SEASON_END_YEARS), max(SEASON_END_YEARS)))
MODEL_DIR <- FINAL_DATA_DIR
# MODEL_DIR <- file.path(PARENT_FINAL_DATA_DIR, '8', '2020-2023')
dir_create(FINAL_DATA_DIR)
dir_create(MODEL_DIR)

MODEL_ID_COLS <- c(
  'competition_id',
  'season_id',
  'game_id',
  'action_id',
  'team_id',
  'period_id'
)

import_parquet <- function(name) {
  read_parquet(
    file.path(FINAL_DATA_DIR, paste0(name, '.parquet'))
  )
}

export_parquet <- function(df, name = deparse(substitute(df))) {
  write_parquet(
    df,
    file.path(FINAL_DATA_DIR, paste0(name, '.parquet'))
  )
}
