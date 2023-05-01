library(dplyr)
library(tidyr)

source(file.path('R', 'helpers.R'))

av <- import_parquet('av')
ava <- import_parquet('ava')
preds_atomic <- import_parquet('preds_atomic')

c(
  'av',
  'ava',
  'preds_atomic',
  'teams',
  'players'
) |> 
  walk(
    ~{
      res <- import_parquet(.x)
      assign(value = res, x = .x, envir = .GlobalEnv)
    }
  )

vaep_atomic <- ava |>
  ## Dummy col so that we don't overwrite the actual col
  mutate(dummy_original_event_id = original_event_id) |> 
  fill(dummy_original_event_id, .direction = 'downup') |> 
  ## TODO: Use atomic_action_id name earlier?
  rename(atomic_action_id = action_id) |> 
  left_join(
    av |>
      ## fill the column so that we can get a 1-to-1 match with the atomic actions
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
    by = join_by(game_id, atomic_action_id == action_id, period_id, team_id)
  )
export_parquet(vaep_atomic)

## debug ----
player_games <- players |> 
  inner_join(
    games |> select(competition_id, season_id, game_id),
    by = join_by(game_id)
  )

player_teams <- player_games |> 
  group_by(competition_id, season_id, player_id, team_id) |> 
  summarize(
    across(minutes_played, sum)
  ) |> 
  ungroup() |> 
  group_by(competition_id, season_id, player_id) |> 
  filter(row_number(desc(minutes_played)) == 1L) |> 
  ungroup() |> 
  inner_join(
    teams,
    by = join_by(team_id)
  )

players_agg <- player_games |> 
  group_by(competition_id, season_id, player_id, player_name) |> 
  summarize(
    n_teams = n_distinct(team_id),
    n_matches = n(),
    n_starts = sum(is_starter),
    across(minutes_played, sum)
  ) |> 
  ungroup() |> 
  left_join(
    player_teams |> 
      select(
        competition_id,
        season_id,
        player_id,
        team_id,
        team_name,
        team_minutes_played = minutes_played
      ),
    by = join_by(competition_id, season_id, player_id)
  )

vaep_atomic |> 
  # filter(season_id == 2023L) |> 
  group_by(competition_id, season_id, player_id) |> 
  summarize(
    n_actions_atomic = n(),
    across(matches('vaep'), sum)
  ) |> 
  ungroup() |> 
  arrange(desc(vaep)) |> 
  inner_join(
    players_agg,
    by = join_by(competition_id, season_id, player_id)
  ) |> 
  mutate(
    across(matches('vaep'), list(p90 = ~.x * 90 / minutes_played))
  ) |> 
  arrange(desc(vaep_p90)) |> 
  filter(season_id == 2022) |> 
  select(
    competition_id,
    season_id,
    player_id,
    player_name,
    team_id,
    team_name,
    n_actions_atomic,
    minutes_played,
    ovaep,
    dvaep,
    vaep,
    ovaep_p90,
    dvaep_p90,
    vaep_p90
  )
