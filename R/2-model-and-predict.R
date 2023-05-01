library(dplyr)
library(tidyr)
library(readr)
library(parsnip)
library(recipes)
library(workflows)
library(parallel)
library(doParallel)
library(xgboost)
library(purrr)

source(file.path('R', 'helpers.R'))

add_possession_cols <- function(df) {
  poss_changes <- df |> 
    arrange(game_id, action_id) |> 
    mutate(across(action_id, list(lag1 = lag))) |> 
    mutate(
      poss_change = case_when(
        # for some reason I'm having to create this column explicitly, instead of just doing lag(action_id)
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
      left_join(
        poss_ids |> 
          select(
            competition_id,
            season_id,
            game_id,
            action_id,
            poss_change,
            possession_id
          )
      ) |>
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

import_xy <- function(suffix = '', games) {
  add_suffix <- function(prefix) {
    sprintf('%s%s%s', prefix, ifelse(suffix != '', '_', ''), suffix)
  }
  inner_join(
    import_parquet(add_suffix('y')),
    import_parquet(add_suffix('x')) |> select(-matches('[1-3]$')),
    by = join_by(game_id, action_id)
  ) |> 
    inner_join(
      import_parquet(add_suffix('actions')) |>
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
    add_possession_cols() |> 
    mutate(
      across(c(scores, concedes), ~ifelse(.x, 'yes', 'no') |> factor()),
      across(where(is.logical), as.integer),
      game_possession_id = sprintf('%s-%s', game_id, possession_id)
    )
}

split_train_test <- function(df) {
  game_ids_train <- games |> filter(season_id < TEST_SEASON_ID) |> pull(game_id)
  game_ids_test <- games |> filter(season_id == TEST_SEASON_ID) |> pull(game_id)
  train <- df |> filter(game_id %in% game_ids_train)
  test <- df |> filter(game_id %in% game_ids_test)
  # withr::local_seed(42)
  # folds <-  group_vfold_cv(trn, group = 'game_possession_id', v = 5)
  list(
    # folds = folds,
    train = train, 
    test = test
  )
}

fit_model <- function(split, target, overwrite = FALSE) {
  path <- file.path(FINAL_DATA_DIR, paste0('model_', target, '.rds'))
  if (file.exists(path) & isFALSE(overwrite)) {
    return(read_rds(path))
  }
  other_target <- setdiff(c('scores', 'concedes'), target)
  rec <- recipe(
    as.formula(sprintf('%s ~ .', target)),
    split$train |> select(-all_of(other_target))
  ) |> 
    update_role(
      all_of(MODEL_ID_COLS),
      new_role = 'id'
    )
  
  spec <- boost_tree(
    trees = 100,
    learn_rate = 0.1,
    tree_depth = 3
  ) |>
    set_mode('classification') |>
    set_engine('xgboost', verbosity = 2)
  
  wf <- workflow(
    preprocessor = rec,
    spec = spec
  )
  
  n_cores <- detectCores()
  cores_for_parallel <- ceiling(n_cores * 0.5)
  cl <- makeCluster(cores_for_parallel)
  registerDoParallel(cl)
  model <- fit(wf, split$train)
  stopCluster(cl)
  
  write_rds(model, path)
  model
}

fit_models <- function(split, overwrite = FALSE) {
  list(
    'scores' = fit_model(split, target = 'scores', overwrite = overwrite),
    'concedes' = fit_model(split, target = 'concedes', overwrite = overwrite)
  )
}

predict_vaep <- function(fits, split) {
  map_dfr(
    c(
      'train',
      'test'
    ),
    ~{
      ovaep <- predict(
        fits_atomic$scores, split[[.x]], type = 'prob'
      ) |> 
        select(ovaep = .pred_yes)
      dvaep <- predict(
        fits_atomic$concedes, split[[.x]], type = 'prob'
      ) |> 
        select(dvaep = .pred_yes)
      
      vaep <- bind_cols(
        ovaep,
        dvaep
      ) |> 
        mutate(vaep = ovaep - dvaep)
      
      bind_cols(
        split[[.x]] |> select(all_of(MODEL_ID_COLS)),
        vaep
      )
    }
  )
}

# df <- import_df() ## for non-atomic
games <- import_parquet('games')
xy_atomic <- import_xy('atomic', games = games)

split_atomic <- split_train_test(xy_atomic)

fits_atomic <- fit_models(split_atomic, overwrite = TRUE)
preds_atomic <- predict_vaep(fits_atomic, split_atomic)
export_parquet(preds_atomic)

## debug ----
preds_atomic |> arrange(desc(vaep))
preds_atomic |> 
  slice_sample(n = 10000) |> 
  pull(vaep) |> 
  hist()
preds_atomic |> filter(is.na(vaep))

