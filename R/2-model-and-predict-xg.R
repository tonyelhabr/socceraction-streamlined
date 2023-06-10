library(dplyr)
library(tidyr)
library(readr)
library(xgboost)
library(purrr)
library(rlang)
library(withr)
library(forcats)

source(file.path('R', 'helpers.R'))

xy_xg <- inner_join(
  import_parquet('y'),
  import_parquet('x') |> select(-matches('_a[2-9]$')),
  by = join_by(game_id, action_id)
) |> 
  inner_join(
    import_parquet('actions') |>
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
games <- import_parquet('games')
game_ids_train <- games |> filter(season_id < TEST_SEASON_ID) |> pull(game_id)
game_ids_test <- games |> filter(season_id == TEST_SEASON_ID) |> pull(game_id)

split <- list(
  train = xy_xg |> filter(game_id %in% game_ids_train), 
  test = xy_xg |> filter(game_id %in% game_ids_test)
)

x_trn <- split$train |> 
  select(
    all_of(
      c(
        'type_pass_a1',
        'type_cross_a1',
        'type_throw_in_a1',
        'type_freekick_crossed_a1',
        'type_freekick_short_a1',
        'type_corner_crossed_a1',
        'type_corner_short_a1',
        'type_take_on_a1',
        'type_foul_a1',
        'type_tackle_a1',
        'type_interception_a1',
        'type_shot_a1',
        'type_shot_penalty_a1',
        'type_shot_freekick_a1',
        'type_keeper_save_a1',
        'type_keeper_claim_a1',
        'type_keeper_punch_a1',
        'type_keeper_pick_up_a1',
        'type_clearance_a1',
        'type_bad_touch_a1',
        'type_non_action_a1',
        'type_dribble_a1',
        'type_goalkick_a1',
        'bodypart_foot_a0',
        'bodypart_head_a0',
        'bodypart_other_a0',
        'bodypart_head/other_a0',
        'bodypart_foot_a1',
        'bodypart_head_a1',
        'bodypart_other_a1',
        'bodypart_head/other_a1',
        'start_x_a0',
        'start_y_a0',
        'start_x_a1',
        'start_y_a1',
        'dx_a1',
        'dy_a1',
        'movement_a1',
        'dx_a01',
        'dy_a01',
        'mov_a01',
        'start_dist_to_goal_a0',
        'start_angle_to_goal_a0',
        'start_dist_to_goal_a1',
        'start_angle_to_goal_a1'
      )
    )
  )
