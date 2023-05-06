library(dplyr)
library(tibble)
library(purrr)

## https://github.com/ML-KULeuven/socceraction/blob/master/socceraction/vaep/formula.py
## https://github.com/ML-KULeuven/socceraction/blob/master/socceraction/atomic/vaep/formula.py
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

get_vaep_atomic_prev_actions <- function(
    team_id,
    type_name,
    concedes,
    scores,
    pov = 'offense'
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
  
  prev_goal_idx <- lag(type_name) %in% c('goal', 'owngoal')
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
  prev_actions <- get_vaep_prev_actions(pov = 'defense', concedes = concedes, ...)
  coalesce(-(concedes - prev_actions), 0)
}

get_ovaep_atomic_value <- function(scores, ...) {
  prev_actions <- get_vaep_atomic_prev_actions(pov = 'offense', scores = scores, ...)
  coalesce(scores - prev_actions, 0)
}

get_dvaep_atomic_value <- function(concedes, ...) {
  prev_actions <- get_vaep_atomic_prev_actions(pov = 'defense', concedes = concedes, ...)
  coalesce(-(concedes - prev_actions), 0)
}


get_vaep_values <- function(scores, concedes, ...) {
  tibble(
    ovaep = get_ovaep_value(
      scores = scores, 
      concedes = concedes,
      ...
    ),
    dvaep = get_dvaep_value(
      scores = scores, 
      concedes = concedes,
      ...
    ),
    vaep = ovaep + dvaep
  )
}

get_vaep_atomic_values <- function(scores, concedes, ...) {
  tibble(
    ovaep_atomic = get_ovaep_atomic_value(
      scores = scores, 
      concedes = concedes,
      ...
    ),
    dvaep_atomic = get_dvaep_atomic_value(
      scores = scores, 
      concedes = concedes,
      ...
    ),
    vaep_atomic = ovaep_atomic + dvaep_atomic
  )
}

## main ----
c(
  'av',
  'ava',
  'preds',
  'preds_atomic'
) |> 
  walk(
    ~{
      res <- import_parquet(.x)
      assign(value = res, x = .x, envir = .GlobalEnv)
    }
  )

av_and_preds <- av |> 
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
        pred_scores,
        pred_concedes
      ),
    by = join_by(!!!MODEL_ID_COLS)
  )

vaep <- av_and_preds |> 
  group_split(game_id, period_id) |> 
  map_dfr(
    ~{
      bind_cols(
        .x |> select(in_test_set, all_of(MODEL_ID_COLS)),
        get_vaep_values(
          team_id = .x$team_id, 
          time_seconds = .x$time_seconds, 
          type_name = .x$type_name, 
          result_name = .x$result_name, 
          concedes = .x$pred_concedes,
          scores = .x$pred_scores
        )
      )
    }
  )

ava_and_preds_atomic <- ava |> 
  select(
    all_of(MODEL_ID_COLS),
    team_id,
    type_name
  ) |> 
  left_join(
    preds_atomic |> 
      select(
        in_test_set,
        any_of(MODEL_ID_COLS),
        pred_scores,
        pred_concedes
      ),
    by = join_by(!!!MODEL_ID_COLS)
  )

vaep_atomic <- ava_and_preds_atomic |> 
  group_split(game_id, period_id) |> 
  map_dfr(
    ~{
      bind_cols(
        .x |> select(in_test_set, all_of(MODEL_ID_COLS)),
        get_vaep_atomic_values(
          team_id = .x$team_id, 
          type_name = .x$type_name, 
          concedes = .x$pred_concedes,
          scores = .x$pred_scores
        )
      )
    }
  )

export_parquet(vaep)
export_parquet(vaep_atomic)
