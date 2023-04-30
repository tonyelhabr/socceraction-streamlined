library(readr)
library(tidymodels)
library(withr)
library(parallel)
library(doParallel)
library(xgboost)
library(arrow)

n_cores <- detectCores()
cores_for_parallel <- ceiling(n_cores * 0.5)

import_parquet <- function(x, dir = 'data/final') {
  read_parquet(
    file.path(dir, paste0(x, '.parquet'))
  )
}

add_possession_cols <- function(df) {
  poss_changes <- df |> 
    arrange(game_id, action_id) |> 
    mutate(across(action_id, list(lag1 = lag))) |> 
    mutate(
      poss_change = case_when(
        # for some reason i'm having to create this column explicitly, instead of just doing lag(action_id)
        is.na(action_id_lag1) ~ TRUE, # for first record in data set
        team_id != lag(team_id) ~ TRUE, 
        game_id != lag(game_id) ~ TRUE,
        period_id != lag(period_id) ~ TRUE,
        TRUE ~ FALSE
      ) 
    ) |> 
    select(-action_id_lag1) |> 
    relocate(poss_change, matches('lag1$'))
  
  poss_ids <- poss_changes |> 
    filter(poss_change) |> 
    group_by(game_id) |> 
    mutate(possession_id = row_number()) |> 
    ungroup()
  
  suppressMessages(
    poss_changes |> 
      left_join(poss_ids |> select(game_id, action_id, poss_change, possession_id)) |> 
      fill(possession_id) |> 
      group_by(game_id, possession_id) |> 
      mutate(
        within_possession_id = row_number()
      ) |> 
      ungroup() |> 
      select(-poss_change) |> 
      relocate(matches('possession_id'))
  )
}

import_df <- function(suffix = '') {
  add_suffix <- function(prefix) {
    sprintf('%s%s%s', prefix, ifelse(suffix != '', '_', ''), suffix)
  }
  inner_join(
    import_parquet(add_suffix('y')),
    import_parquet(add_suffix('x')) |> select(-matches('[1-3]$')),
    by = c('game_id', 'action_id')
  ) |> 
    inner_join(
      import_parquet('actions') |> select(game_id, team_id, period_id, action_id),
      by = c('game_id', 'action_id')
    ) |> 
    add_possession_cols() |> 
    mutate(
      across(c(scores, concedes), ~ifelse(.x, 'yes', 'no') |> factor()),
      across(where(is.logical), as.integer),
      game_possession_id = sprintf('%s-%s', game_id, possession_id)
    )
}

split_trn_tst <- function(df) {
  trn <- df |> filter(game_id %in% game_ids_trn)
  tst <- df |> filter(game_id %in% game_ids_tst)
  withr::local_seed(42)
  folds <-  group_vfold_cv(trn, group = 'game_possession_id', v = 5)
  list(
    train = trn, 
    test = tst,
    folds = folds
  )
}

fit_model <- function(split, target) {
  other_target <- setdiff(c('scores', 'concedes'), target)
  rec <- recipe(
    as.formula(sprintf('%s ~ .', target)),
    split$train |> select(-.data[[other_target]])
  ) |> 
    update_role(
      game_id,
      action_id,
      team_id,
      period_id,
      possession_id,
      within_possession_id,
      game_possession_id,
      new_role = 'id'
    )
  
  spec <- boost_tree(
  ) |>
    set_mode('classification') |>
    set_engine('xgboost', verbosity = 2)
  
  wf <- workflow(
    preprocessor = rec,
    spec = spec
  )

  cl <- makeCluster(cores_for_parallel)
  registerDoParallel(cl)
  
  model <- fit(wf, split$train)
  
  stopCluster(cl)
  
  model
}

fit_models <- function(split) {
  list(
    'scores' = fit_model(split, 'scores'),
    'concedes' = fit_model(split, 'concedes')
  )
}

# df <- import_df()
df_atomic <- import_df('atomic')
games <- import_parquet('games')
game_ids_trn <- games |> filter(season_id < 2023) |> pull(game_id)
game_ids_tst <- games |> filter(season_id == 2023) |> pull(game_id)
# split <- df |> split_trn_tst()
# split_atomic <- df_atomic |> split_trn_tst()
split_atomic <- list(
  train = df_atomic |> filter(game_id %in% game_ids_trn),
  test = df_atomic |> filter(game_id %in% game_ids_tst)
)

fits_atomic <- split_atomic |> fit_models()
