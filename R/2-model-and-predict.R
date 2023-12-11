library(dplyr)
library(tidyr)
library(readr)
library(xgboost)
library(purrr)
library(rlang)
library(withr)
library(forcats)

source(file.path('R', 'helpers.R'))

convert_atomic_bool_to_suffix <- function(atomic = TRUE) {
  ifelse(isTRUE(atomic), '_atomic', '')
}

MODEL_COLS <- list(
  ## notebooks also have type_id_a0 and result_id_a0, but those are unncessary since we onehot type and result
  'non-atomic' = c('type_pass_a0', 'type_cross_a0', 'type_throw_in_a0', 'type_freekick_crossed_a0', 'type_freekick_short_a0', 'type_corner_crossed_a0', 'type_corner_short_a0', 'type_take_on_a0', 'type_foul_a0', 'type_tackle_a0', 'type_interception_a0', 'type_shot_a0', 'type_shot_penalty_a0', 'type_shot_freekick_a0', 'type_keeper_save_a0', 'type_keeper_claim_a0', 'type_keeper_punch_a0', 'type_keeper_pick_up_a0', 'type_clearance_a0', 'type_bad_touch_a0', 'type_non_action_a0', 'type_dribble_a0', 'type_goalkick_a0', 'bodypart_foot_a0', 'bodypart_head_a0', 'bodypart_other_a0', 'bodypart_head/other_a0', 'result_fail_a0', 'result_success_a0', 'result_offside_a0', 'result_owngoal_a0', 'result_yellow_card_a0', 'result_red_card_a0', 'goalscore_team', 'goalscore_opponent', 'goalscore_diff', 'start_x_a0', 'start_y_a0', 'end_x_a0', 'end_y_a0', 'dx_a0', 'dy_a0', 'movement_a0', 'start_dist_to_goal_a0', 'start_angle_to_goal_a0', 'end_dist_to_goal_a0', 'end_angle_to_goal_a0', 'period_id_a0', 'time_seconds_a0', 'time_seconds_overall_a0'),
  'atomic' = c('type_pass_a0', 'type_cross_a0', 'type_throw_in_a0', 'type_freekick_crossed_a0', 'type_freekick_short_a0', 'type_corner_crossed_a0', 'type_corner_short_a0', 'type_take_on_a0', 'type_foul_a0', 'type_tackle_a0', 'type_interception_a0', 'type_shot_a0', 'type_shot_penalty_a0', 'type_shot_freekick_a0', 'type_keeper_save_a0', 'type_keeper_claim_a0', 'type_keeper_punch_a0', 'type_keeper_pick_up_a0', 'type_clearance_a0', 'type_bad_touch_a0', 'type_non_action_a0', 'type_dribble_a0', 'type_goalkick_a0', 'type_receival_a0', 'type_out_a0', 'type_offside_a0', 'type_goal_a0', 'type_owngoal_a0', 'type_yellow_card_a0', 'type_red_card_a0', 'type_corner_a0', 'type_freekick_a0', 'bodypart_foot_a0', 'bodypart_head_a0', 'bodypart_other_a0', 'bodypart_head/other_a0', 'goalscore_team', 'goalscore_opponent', 'goalscore_diff', 'x_a0', 'y_a0', 'dist_to_goal_a0', 'angle_to_goal_a0', 'dx_a0', 'dy_a0', 'period_id_a0', 'time_seconds_a0', 'time_seconds_overall_a0')
)

.select_x <- function(df, atomic = TRUE) {
  df |> 
    select(
      # -any_of(c('scores', 'concedes', 'goal_from_shot', 'goal')), 
      # -all_of(MODEL_ID_COLS)
      all_of(MODEL_COLS[[ifelse(atomic, 'atomic', 'non-atomic')]])
    ) |> 
    df_to_mat()
}

fit_model <- function(split, target, atomic = TRUE, overwrite = FALSE) {
  suffix <- convert_atomic_bool_to_suffix(atomic)
  path <- file.path(MODEL_DIR, paste0('model_', target, suffix, '.model'))
  if (file.exists(path) & isFALSE(overwrite)) {
    return(xgboost::xgb.load(path))
  }
  x <- .select_x(split$train, atomic = atomic)
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
      function(.x) {
        fit_model(
          split, 
          target = .x,
          atomic = atomic,
          overwrite = overwrite
        )
      }
    )
}

.predict_value <- function(fit, df, atomic = TRUE, select_f, ...) {
  x <- select_f(df, atomic = atomic)
  predict(fit, newdata = x, ...)
}

predict_values <- function(fits, split, atomic = FALSE) {
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
      ## if there is no train/test set
      if (nrow( split[[.x]]) == 0) {
        return(tibble())
      }
      pred_scores <- tibble(
        !!col_scores := .predict_value(
          fit = fits$scores,
          df = split[[.x]],
          atomic = atomic,
          select_f = .select_x
        )
      )
      
      pred_concedes <-  tibble(
        !!col_concedes := .predict_value(
          fit = fits$concedes,
          df = split[[.x]],
          atomic = atomic,
          select_f = .select_x
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
          in_test = season_id %in% TEST_SEASON_IDS, 
          .before = 1
        )
    }
  )
}

.summarize_pred_contrib <- function(fit, df, atomic = TRUE, n = 50000, seed = 42) {
  withr::local_seed(seed)
  df <- slice_sample(df, n = n)
  
  contrib <- .predict_value(fit, df, atomic = atomic, predcontrib = TRUE) |>
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

summarize_pred_contrib <- function(fits, df, atomic) {
  list(
    'scores',
    'concedes'
  ) |> 
    map_dfr(
      ~{
        .summarize_pred_contrib(
          fit = fits[[.x]], 
          df = df,
          atomic = atomic
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
xy <- import_xy(games = games)
xy_atomic <- import_xy('atomic', games = games)
# setdiff(colnames(xy), c(MODEL_COLS[['non-atomic']], MODEL_ID_COLS, c('scores', 'concedes', 'goal_from_shot')))
# setdiff(colnames(xy_atomic), c(MODEL_COLS[['atomic']], MODEL_ID_COLS, c('scores', 'concedes', 'goal')))

split <- split_train_test(xy, games = games)
split_atomic <- split_train_test(xy_atomic, games = games)

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

export_parquet(preds)
export_parquet(preds_atomic)

contrib <- summarize_pred_contrib(
  fits = fits,
  df = split$train,
  atomic = FALSE
)

contrib_atomic <- summarize_pred_contrib(
  fits = fits_atomic,
  df = split_atomic$train,
  atomic = TRUE
)

export_parquet(contrib)
export_parquet(contrib_atomic)
