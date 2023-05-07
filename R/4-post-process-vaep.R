library(dplyr)
library(tidyr)
library(purrr)

source(file.path('R', 'helpers.R'))

c(
  'av',
  'ava',
  'vaep',
  'vaep_atomic',
  'teams',
  'players',
  'games'
) |> 
  walk(
    ~{
      res <- import_parquet(.x)
      assign(value = res, x = .x, envir = .GlobalEnv)
    }
  )

## debug ----
# preds_atomic |> arrange(desc(vaep_atomic))
# preds_atomic |> 
#   slice_sample(n = 10000) |> 
#   pull(vaep_atomic) |> 
#   hist()

all_vaep <- ava |>
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
    vaep |> select(-in_test),
    by = join_by(!!!MODEL_ID_COLS)
  ) |> 
  left_join(
    vaep_atomic,
    by = join_by(
      # c(setdiff(MODEL_ID_COLS), 'action_id'),
      competition_id,
      season_id,
      game_id,
      period_id,
      team_id,
      atomic_action_id == action_id
    )
  )
export_parquet(all_vaep)
rm(list = c('av', 'ava', 'vaep', 'vaep_atomic'))
gc()

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
    starts = sum(is_starter),
    across(minutes_played, sum)
  ) |> 
  ungroup() |> 
  left_join(
    player_games |> 
      filter(minutes_played > 0) |> 
      count(competition_id, season_id, player_id, name = 'games_played'),
    by = join_by(competition_id, season_id, player_id)
  ) |> 
  left_join(
    player_teams |> 
      select(
        competition_id,
        season_id,
        player_id,
        team_id,
        team_name
      ),
    by = join_by(competition_id, season_id, player_id)
  )

vaep_by_player_season <- all_vaep |> 
  # filter(season_id == 2023L) |> 
  group_by(in_test, competition_id, season_id, player_id) |> 
  summarize(
    n_actions = sum(!is.na(action_id)),
    n_actions_atomic = n(),
    across(matches('vaep'), \(x) sum(x, na.rm = TRUE))
  ) |> 
  ungroup() |> 
  inner_join(
    players_agg,
    by = join_by(competition_id, season_id, player_id)
  ) |> 
  mutate(
    across(matches('^vaep'), list(p90 = \(x) x * 90 / minutes_played))
  ) |> 
  # filter(season_id == 2021) |> 
  select(
    in_test, 
    competition_id,
    season_id,
    player_id,
    player_name,
    team_id,
    team_name,
    n_actions,
    n_actions_atomic,
    minutes_played,
    games_played,
    vaep,
    vaep_atomic,
    vaep_p90,
    vaep_atomic_p90,
  ) |> 
  arrange(desc(vaep_atomic))
export_parquet(vaep_by_player_season)

vaep_by_player_season |> arrange(desc(vaep_atomic))
vaep_by_player_season |> 
  group_by(in_test, competition_id, season_id) |> 
  summarize(
    across(
      c(
        vaep,
        vaep_atomic,
      ),
      \(x) sum(x, na.rm = TRUE)
    )
  ) |> 
  ungroup() |> 
  arrange(competition_id, season_id)
