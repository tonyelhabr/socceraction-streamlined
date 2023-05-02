library(arrow)
library(fs)

COMPETITION_ID <- 8
SEASON_END_YEARS <- 2021:2023
TEST_SEASON_ID <- max(SEASON_END_YEARS)
PROCESSED_DATA_DIR <- 'data/processed'
FINAL_DATA_DIR <- 'data/final'
dir_create(FINAL_DATA_DIR)

MODEL_ID_COLS <- c(
  'competition_id',
  'season_id',
  'game_id',
  'action_id',
  'team_id',
  'period_id',
  'possession_id',
  'within_possession_id',
  'game_possession_id'
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
