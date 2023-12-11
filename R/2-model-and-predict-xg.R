library(dplyr)
library(tidyr)
library(readr)
library(xgboost)
library(purrr)
library(rlang)
library(withr)
library(forcats)

source(file.path('R', 'helpers.R'))

XG_MODEL_ID_COLS <- c(
  'start_x_a0',
  'start_y_a0',
  'start_dist_to_goal_a0',
  'start_angle_to_goal_a0',
  'bodypart_id_a0'
)

## https://github.com/ML-KULeuven/soccer_xg/blob/master/notebooks/4-creating-custom-xg-pipelines.ipynb
.select_xg_x <- function(df) {
  df |> 
    transmute(
      start_x_a0,
      start_y_a0,
      start_dist_to_goal_a0,
      start_angle_to_goal_a0,
      bodypart_id_a0 = case_when(
        bodypart_foot_a0 == 1L ~ 0L,
        bodypart_head_a0 == 1L ~ 1L,
        bodypart_other_a0 == 1L ~ 2L,
        `bodypart_head/other_a0` == 1L ~ 3L
      )
    ) |> 
    df_to_mat()
}

fit_xg_model <- function(split, overwrite = FALSE) {
  path <- file.path(MODEL_DIR, paste0('model_xg.model'))
  if (file.exists(path) & isFALSE(overwrite)) {
    return(xgboost::xgb.load(path))
  }
  x <- .select_xg_x(split$train)
  y <- as.integer(split$train$scores) - 1L
  fit <- xgboost::xgboost(
    data = x,
    label = y,
    eval_metric = 'logloss',
    early_stopping_round = 10,
    nrounds = 500,
    print_every_n = 10,
    max_depth = 3, 
    n_jobs = -3
  )
  xgboost::xgb.save(fit, path)
  fit
}

.predict_xg <- function(fit, df, ...) {
  x <- .select_xg_x(df)
  predict(fit, newdata = x, ...)
}

predict_xg <- function(fits, split, atomic = TRUE) {
  col_scores <- sym(paste0('pred_scores', suffix, ''))
  map_dfr(
    c(
      'train',
      'test'
    ),
    ~{
      ## if there is no train/test set
      if (nrow( split[[.x]]) == 0) {
        return(tibble())
      }
      pred <- tibble(
        !!col_scores := .predict_xg(
          fit = fits,
          df = split[[.x]]
        )
      )
      
      actual <- bind_cols(
        split[[.x]] |> select(scores),
        pred_scores
      )
      
      bind_cols(
        split[[.x]] |> select(all_of(XG_MODEL_ID_COLS)),
        vaep 
      ) |>
        mutate(
          in_test = season_id %in% TEST_SEASON_IDS, 
          .before = 1
        )
    }
  )
}

## main ----
games <- import_parquet('games')
xy <- import_xy(games = games)
open_play_shots <- filter(
  xy,
  type_shot_a0 == 1
)
split <- split_train_test(open_play_shots, games = games)

# xg_model <- fit_xg_model(
#   split = split,
#   overwrite = TRUE
# )
