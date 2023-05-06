library(dplyr)
library(tidyr)
library(readr)
library(xgboost)
library(purrr)
library(rlang)
library(withr)
library(forcats)

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

.select_x <- function(df) {
  df |> 
    select(-c(scores, concedes), -all_of(MODEL_ID_COLS)) |> 
    df_to_mat()
}

fit_model <- function(split, target, atomic = TRUE, overwrite = FALSE) {
  suffix <- convert_atomic_bool_to_suffix(atomic)
  path <- file.path(FINAL_DATA_DIR, paste0('model_', target, suffix, '_r.model'))
  if (file.exists(path) & isFALSE(overwrite)) {
    return(xgboost::xgb.load(path))
  }
  x <- .select_x(split$train)
  y <- as.integer(split$train[[target]])
  fit <- xgboost::xgboost(
    data = x,
    label = y,
    eval_metric = 'logloss',
    nrounds = 100,
    print_every_n = 10,
    max_depth = 3, 
    n_jobs = -3
  )
  xgboost::xgb.save(fit, path)
  fit
}

fit_models <- function(split, atomic = TRUE, overwrite = FALSE) {
  list(
    'scores',
    'concedes'
  ) |> 
    set_names() |> 
    map(
      ~{
        fit_model(
          split, 
          target = .x,
          atomic = atomic, 
          overwrite = overwrite
        )
      }
    )
}

read_py_model <- function(atomic, target) {
  suffix <- convert_atomic_bool_to_suffix(atomic)
  path <- file.path(FINAL_DATA_DIR, paste0('model_', target, suffix, '.model'))
  xgboost::xgb.load(path)
}

read_py_models <- function(atomic = TRUE) {
  list(
    'scores',
    'concedes'
  ) |> 
    set_names() |> 
    map(
      ~{
        read_py_model(
          atomic = atomic, 
          target = .x
        )
      }
    )
}

.predict_vaep <- function(fit, df, x = NULL) {
  if (is.null(x)) {
    x <- .select_x(df)
  }
  predict(fit, newdata = x)
}

.summarize_vaep_pred_contrib <- function(fit, df, target, atomic = TRUE, n = 10000, seed = 42) {
  withr::local_seed(seed)
  df <- slice_sample(df, n = n)
  x <- .select_x(df)
  suffix <- convert_atomic_bool_to_suffix(atomic)
  col_pred <- sym('.pred')
  # preds <- tibble(
  #   !!col_pred := 1 - .predict_vaep(fit, x = x)
  # )
  
  contrib <- predict(fit, x, predcontrib = TRUE) |>
    as.data.frame() |>
    as_tibble() |>
    rename(baseline = BIAS)
  
  # feature_values <- x |> 
  #   as.data.frame() |> 
  #   mutate(
  #     across(everything(), scale)
  #   ) |> 
  #   pivot_longer(
  #     everything(),
  #     names_to = 'feature',
  #     values_to = 'feature_value'
  #   ) |> 
  #   as_tibble()
  
  long_contrib <- contrib |> 
    pivot_longer(
      -c(baseline),
      names_to = 'feature',
      values_to = 'contrib_value'
    )
  
  long_contrib |>
    group_by(feature) |>
    summarize(
      across(contrib_value, \(x) mean(abs(x))),
    ) |>
    ungroup() |>
    mutate(
      contrib_value_rank = row_number(desc(contrib_value)),
      across(feature, ~fct_reorder(feature, desc(contrib_value_rank)))
    ) |>
    arrange(contrib_value_rank)
}

predict_vaep <- function(fits_r, fits_py, split, atomic = TRUE) {
  suffix <- convert_atomic_bool_to_suffix(atomic)
  col_scores_r <- sym(paste0('pred_scores', suffix, '_r'))
  col_concedes_r <- sym(paste0('pred_concedes', suffix, '_r'))
  col_total_r <- sym(paste0('pred', suffix, '_r'))
  col_scores_py <- sym(paste0('pred_scores', suffix, '_py'))
  col_concedes_py <- sym(paste0('pred_concedes', suffix, '_py'))
  col_total_py <- sym(paste0('pred', suffix, '_py'))
  map_dfr(
    c(
      'train',
      'test'
    ),
    ~{
      pred_scores_r <- tibble(
        !!col_scores_r := 1 - .predict_vaep(
          fits_r$scores,
          split[[.x]]
        )
      )
      
      pred_concedes_r <-  tibble(
        !!col_concedes_r := 1 - .predict_vaep(
          fits_r$concedes,
          split[[.x]]
        )
      )
      
      pred_scores_py <- tibble(
        !!col_scores_py := .predict_vaep(
          fits_py$scores,
          split[[.x]]
        )
      )
      
      pred_concedes_py <- tibble(
        !!col_concedes_py := .predict_vaep(
          fits_py$concedes,
          split[[.x]]
        )
      )
      
      vaep <- bind_cols(
        split[[.x]] |> select(scores),
        pred_scores_r,
        pred_scores_py,
        split[[.x]] |> select(concedes),
        pred_concedes_r,
        pred_concedes_py
      ) |> 
        mutate(
          !!col_total_r := !!col_scores_r - !!col_concedes_r,
          !!col_total_py := !!col_scores_py - !!col_concedes_py
        )
      
      bind_cols(
        split[[.x]] |> select(all_of(MODEL_ID_COLS)),
        vaep 
      ) |>
        mutate(
          in_test = season_id == TEST_SEASON_ID, 
          .before = 1
        )
    }
  )
}

games <- import_parquet('games')
xy <- import_xy( games = games)
xy_atomic <- import_xy('atomic', games = games)

split <- split_train_test(xy)
split_atomic <- split_train_test(xy_atomic)

fits_r <- fit_models(split = split, atomic = FALSE, overwrite = TRUE)
fits_atomic_r <- fit_models(split = split_atomic, atomic = TRUE, overwrite = TRUE)

fits_py <- read_py_models(atomic = FALSE)
fits_atomic_py <- read_py_models(atomic = TRUE)

preds <- predict_vaep(
  fits_r = fits_r, 
  fits_py = fits_py, 
  split = split, 
  atomic = FALSE
)
preds_atomic <- predict_vaep(
  fits_r = fits_atomic_r, 
  fits_py = fits_atomic_py, 
  split = split_atomic, 
  atomic = TRUE
)

export_parquet(preds)
export_parquet(preds_atomic)

## debug ----
# preds_atomic |> 
#   mutate(
#     d = vaep_atomic_r - vaep_atomic_py
#   ) |> 
#   arrange(desc(d))
