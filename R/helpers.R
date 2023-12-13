library(arrow)
library(fs)

COMPETITION_ID <- 8
SEASON_END_YEARS <- c(2013:2019, 2021:2022)
TEST_SEASON_IDS <- c(2018:2019, 2021:2022) # SEASON_END_YEARS # max(SEASON_END_YEARS)
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

add_suffix <- function(prefix, suffix) {
  sprintf('%s%s%s', prefix, ifelse(suffix != '', '_', ''), suffix)
}

import_xy <- function(suffix = '', games) {
  inner_join(
    import_parquet(add_suffix('y', suffix = suffix)),
    import_parquet(add_suffix('x', suffix = suffix)) |> select(-matches('_a[1-2]$')),
    by = join_by(game_id, action_id)
  ) |> 
    inner_join(
      import_parquet(add_suffix('actions', suffix = suffix)) |>
        select(
          game_id,
          team_id,
          period_id,
          action_id
        ),
      by = join_by(game_id, action_id)
    ) |>
    inner_join(
      games |> select(competition_id, season_id, game_id),
      by = join_by(game_id)
    ) |> 
    mutate(
      across(c(scores, concedes), ~ifelse(.x, 'yes', 'no') |> factor()),
      across(where(is.logical), as.integer)
    )
}

df_to_mat <- function(df) {
  model.matrix(
    ~.+0,
    data = model.frame(
      ~.+0,
      df,
      na.action = na.pass
    )
  )
}

split_train_test <- function(df, games) {
  game_ids_train <- games |> filter(!(season_id %in% TEST_SEASON_IDS)) |> pull(game_id)
  game_ids_test <- games |> filter(season_id %in% TEST_SEASON_IDS) |> pull(game_id)
  train <- df |> filter(game_id %in% game_ids_train)
  test <- df |> filter(game_id %in% game_ids_test)
  list(
    train = train, 
    test = test
  )
}
