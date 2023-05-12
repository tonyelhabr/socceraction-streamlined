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
  path <- file.path(FINAL_DATA_DIR, paste0('model_', target, suffix, '.model'))
  if (file.exists(path) & isFALSE(overwrite)) {
    return(xgboost::xgb.load(path))
  }
  x <- .select_x(split$train)
  y <- as.integer(split$train[[target]]) - 1L
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

.predict_value <- function(fit, df, ...) {
  x <- .select_x(df)
  predict(fit, newdata = x, ...)
}

predict_values <- function(fits, split, atomic = TRUE) {
  suffix <- convert_atomic_bool_to_suffix(atomic)
  col_scores <- sym(paste0('pred_scores', suffix, ''))
  col_concedes <- sym(paste0('pred_concedes', suffix, ''))
  col_total <- sym(paste0('pred', suffix, ''))
  map_dfr(
    c(
      'train',
      'test'
    ),
    ~{
      pred_scores <- tibble(
        !!col_scores := .predict_value(
          fits$scores,
          split[[.x]]
        )
      )
      
      pred_concedes <-  tibble(
        !!col_concedes := .predict_value(
          fits$concedes,
          split[[.x]]
        )
      )

      vaep <- bind_cols(
        split[[.x]] |> select(scores),
        pred_scores,
        split[[.x]] |> select(concedes),
        pred_concedes
      ) |> 
        mutate(
          !!col_total := !!col_scores - !!col_concedes
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

.summarize_pred_contrib <- function(fit, df, n = 50000, seed = 42) {
  withr::local_seed(seed)
  df <- slice_sample(df, n = n)

  contrib <- .predict_value(fit, df, predcontrib = TRUE) |>
    as.data.frame() |>
    as_tibble() |>
    rename(baseline = BIAS)
  
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

summarize_pred_contrib <- function(fits, df) {
  list(
    'scores',
    'concedes'
  ) |> 
    map_dfr(
      ~{
        .summarize_pred_contrib(
          fit = fits[[.x]], 
          df = df
        ) |> 
          mutate(
            side = .x,
            .before = 1
          )
      }
    )
}

## main ----
games <- import_parquet('games')
xy <- import_xy( games = games)
xy_atomic <- import_xy('atomic', games = games)

split <- split_train_test(xy)
split_atomic <- split_train_test(xy_atomic)

fits <- fit_models(
  split = split, 
  atomic = FALSE, 
  overwrite = FALSE
)
fits_atomic <- fit_models(
  split = split_atomic, 
  atomic = TRUE,
  overwrite = FALSE
)

preds <- predict_values(
  fits = fits, 
  split = split, 
  atomic = FALSE
)
preds_atomic <- predict_values(
  fits = fits_atomic, 
  split = split_atomic, 
  atomic = TRUE
)

contrib <- summarize_pred_contrib(
  fits = fits,
  df = split$test
)

contrib_atomic <- summarize_pred_contrib(
  fits = fits_atomic,
  df = split_atomic$test
)

export_parquet(preds)
export_parquet(preds_atomic)
export_parquet(contrib)
export_parquet(contrib_atomic)
