library(dplyr)
library(tibble)
library(purrr)

get_vaep_prev_actions <- function(
    team_id,
    time_seconds,
    type_name,
    result_name,
    concedes,
    scores,
    pov = 'offense',
    same_phase_seconds = 10
) {
  same_team <- lag(team_id) == team_id
  if (pov == 'offense') {
    pos <- scores
    neg <- concedes
  } else if (pov == 'defense') {
    pos <- concedes
    neg <- scores
  }
  prev_actions <- lag(pos) * same_team + lag(neg) * !same_team
  too_long_idx <- abs(time_seconds - lag(time_seconds)) > same_phase_seconds
  prev_actions[too_long_idx] <- 0
  
  prev_goal_idx <- (lag(type_name) %in% c('shot', 'shot_freekick', 'shot_penalty')) & (lag(result_name) == 'success')
  prev_actions[prev_goal_idx] <- 0
  prev_actions
}

get_ovaep_value <- function(type_name, scores, ...) {
  prev_actions <- get_vaep_prev_actions(pov = 'offense', type_name = type_name, scores = scores, ...)
  penalty_idx <- type_name == 'shot_penalty'
  prev_actions[penalty_idx] <- 0.792453
  
  # fixed odds of scoring when corner
  corner_idx <- type_name %in% c('corner_crossed', 'corner_short')
  prev_actions[corner_idx] <- 0.046500
  
  coalesce(scores - prev_actions, 0)
}

get_dvaep_value <- function(concedes, ...) {
  prev_actions <- get_vaep_value(pov = 'defense', concedes = concedes, ...)
  coalesce(-(concedes - prev_actions), 0)
}

get_vaep_values <- function(scores, concedes, ...) {
  tibble(
    offensive_value = get_ovaep_value(
      scores = scores, 
      concedes = concedes,
      ...
    ),
    defensive_value = get_dvaep_value(
      scores = scores, 
      concedes = concedes,
      ...
    ),
    value = offensive_value + defensive_value
  )
}

convert_preds_to_vaep <- function(actions, preds) {
  actions_and_preds <- actions |> 
    select(
      all_of(MODEL_ID_COLS),
      team_id,
      time_seconds, 
      type_name, 
      result_name
    ) |> 
    left_join(
      preds |> 
        select(
          in_test_set,
          any_of(MODEL_ID_COLS),
          pred_scores_r,
          pred_concedes_r
        ),
      by = join_by(!!!MODEL_ID_COLS)
    )
  
  actions_and_preds |> 
    group_split(game_id) |> 
    map_dfr(
      ~{
        bind_cols(
          .x |> select(in_test_set, all_of(MODEL_ID_COLS)),
          get_vaep_values(
            team_id = .x$team_id, 
            time_seconds = .x$time_seconds, 
            type_name = .x$type_name, 
            result_name = .x$result_name, 
            concedes = .x$pred_concedes_r,
            scores = .x$pred_scores_r
          )
        )
      }
    )
}

convert_preds_to_vaep(
  import_parquet('av'),
  import_parquet('preds')
)

convert_preds_to_vaep(
  import_parquet('av'),
  import_parquet('preds')
)

value(av[1:100, ], preds$pred_scores_r[1:100], preds$pred_concedes_r[1:100])

values <- value(av, preds$pred_scores_r, preds$pred_concedes_py)

