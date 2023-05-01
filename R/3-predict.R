library(dplyr)
library(tidyr)
library(purrr)

source(file.path('R', 'helpers.R'))

av <- import_parquet('av')
ava <- import_parquet('ava')

vaep_atomic <- ava |>
  mutate(dummy_original_event_id = original_event_id) |> 
  fill(dummy_original_event_id, .direction = 'downup') |> 
  rename(atomic_action_id = action_id) |> 
  left_join(
    av |>
      mutate(dummy_original_event_id = original_event_id) |>
      fill(dummy_original_event_id, .direction = 'downup') |>
      select(
        action_id,
        game_id,
        dummy_original_event_id,
        time_seconds,
        period_id,
        team_id,
        player_id,
        type_id,
        result_id,
        result_name,
        xt
      ),
    by = join_by(
      game_id,
      dummy_original_event_id,
      period_id,
      time_seconds,
      team_id,
      player_id,
      type_id
    )
  ) |>
  select(-dummy_original_event_id) |>
  relocate(
    action_id,
    .after = atomic_action_id
  ) |> 
  left_join(
    preds_atomic,
    by = join_by(game_id, action_id, period_id, team_id)
  )
export_parquet(vaep_atomic)
