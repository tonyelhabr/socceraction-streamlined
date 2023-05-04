library(dplyr)
library(tidyr)
library(readr)
library(parsnip)
library(recipes)
library(workflows)
library(xgboost)
library(purrr)
library(butcher)
library(rlang)

source(file.path('R', 'helpers.R'))

add_suffix <- function(prefix, suffix) {
  sprintf('%s%s%s', prefix, ifelse(suffix != '', '_', ''), suffix)
}

import_xy <- function(suffix = '', games) {
  inner_join(
    import_parquet(add_suffix('y', suffix = suffix)),
    import_parquet(add_suffix('x', suffix = suffix)),
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

split_train_test <- function(df) {
  game_ids_train <- games |> filter(season_id < TEST_SEASON_ID) |> pull(game_id)
  game_ids_test <- games |> filter(season_id == TEST_SEASON_ID) |> pull(game_id)
  train <- df |> filter(game_id %in% game_ids_train)
  test <- df |> filter(game_id %in% game_ids_test)
  list(
    train = train, 
    test = test
  )
}

convert_atomic_bool_to_suffix <- function(atomic = TRUE) {
  ifelse(isTRUE(atomic), '_atomic', '')
}

fit_model <- function(split, target, atomic = TRUE, overwrite = FALSE) {
  suffix <- convert_atomic_bool_to_suffix(atomic)
  path <- file.path(FINAL_DATA_DIR, paste0('model_', target, suffix, '.rds'))
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
  
  # spec <- boost_tree(
  #   trees = 50,
  #   # learn_rate = 0.1,
  #   tree_depth = 3
  # ) |>
  #   set_mode('classification') |>
  #   set_engine('xgboost')
  
  spec <- boost_tree(
    mode = 'classification',
    trees = 50,
    tree_depth = 3,
    mtry = NULL,
    learn_rate = NULL,
    min_n = NULL,
    loss_reduction = NULL,
    sample_size = NULL,
    stop_iter = NULL
  ) |> 
    set_engine(
      'xgboost',
      stop_window = NULL,
      stop_val = NULL,
      nthread = -3,
      verbose = 1
    )
  
  wf <- workflow(
    preprocessor = rec,
    spec = spec
  )

  model <- fit(wf, split$train)

  suffix <- convert_atomic_bool_to_suffix(atomic)
  xgboost::xgb.save(model$fit$fit$fit, file.path(FINAL_DATA_DIR, paste0('model_', target, suffix, '_r.model')))
  model <- butcher(model)
  write_rds(model, path)
  model
}

fit_models <- function(split, atomic = TRUE, overwrite = FALSE) {
  list(
    'scores' = fit_model(split, target = 'scores', atomic = atomic, overwrite = overwrite),
    'concedes' = fit_model(split, target = 'concedes', atomic = atomic, overwrite = overwrite)
  )
}

predict_vaep <- function(fits, split, atomic = TRUE) {
  suffix <- convert_atomic_bool_to_suffix(atomic)
  col_o <- sym(paste0('ovaep', suffix))
  col_d <- sym(paste0('dvaep', suffix))
  col_total <- sym(paste0('vaep', suffix))
  map_dfr(
    c(
      'train',
      'test'
    ),
    ~{
      ovaep <- predict(
        fits$scores, split[[.x]], type = 'prob'
      ) |> 
        select(!!col_o := .pred_yes)
      dvaep <- predict(
        fits$concedes, split[[.x]], type = 'prob'
      ) |> 
        select(!!col_d := .pred_yes)

      vaep <- bind_cols(
        split[[.x]] |> select(scores),
        ovaep,
        split[[.x]] |> select(concedes),
        dvaep
      ) |> 
        mutate(!!col_total := !!col_o - !!col_d)
      
      bind_cols(
        split[[.x]] |> select(all_of(MODEL_ID_COLS)),
        vaep 
      ) |>
        mutate(in_test = season_id == TEST_SEASON_ID, .before = 1)
    }
  )
}

games <- import_parquet('games')
xy <- import_xy( games = games)
xy_atomic <- import_xy('atomic', games = games)

split <- split_train_test(xy)
split_atomic <- split_train_test(xy_atomic)

fits <- fit_models(split = split, atomic = FALSE, overwrite = TRUE)
fits_atomic <- fit_models(split = split_atomic, atomic = TRUE, overwrite = TRUE)

preds <- predict_vaep(fits, split = split, atomic = FALSE)
preds_atomic <- predict_vaep(fits_atomic, split = split_atomic, atomic = TRUE)

export_parquet(preds)
export_parquet(preds_atomic)
