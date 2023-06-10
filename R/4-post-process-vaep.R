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
  # mutate(
  #   g_atomic = as.integer(type_name %in% c('goal', 'owngoal'))
  # ) |> 
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
      transmute(
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
        g = as.integer((type_name %in% c('shot', 'shot_freekick', 'shot_penalty')) & (result_name == 'success')),
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

## debug ----
player_games <- players |> 
  inner_join(
    games |> select(competition_id, season_id, game_id),
    by = join_by(game_id)
  )

player_starting_positions <- player_games |> 
  group_by(competition_id, season_id, player_id, starting_position) |> 
  summarize(
    across(minutes_played, sum)
  ) |> 
  ungroup()

most_common_player_starting_positions <- player_starting_positions |> 
  filter(starting_position != 'Sub') |> 
  group_by(competition_id, season_id, player_id) |> 
  mutate(
    prop = minutes_played / sum(minutes_played)
  ) |> 
  filter(row_number(desc(minutes_played)) == 1L) |> 
  ungroup()

## TODO: Use this to determine player position weightings?
# player_positions <- player_games |>
#   group_by(competition_id, season_id, player_id) |>
#   summarize(
#     total_minutes_played = sum(minutes_played, na.rm = TRUE)
#   ) |>
#   ungroup() |>
#   left_join(
#     player_starting_positions |>
#       filter(starting_position != 'Sub') |>
#       group_by(competition_id, season_id, player_id) |>
#       mutate(
#         starter_minutes_played = sum(minutes_played),
#         prop = minutes_played / starter_minutes_played
#       ) |>
#         ungroup(),
#     by = join_by(competition_id, season_id, player_id)
#   ) |>
#   filter(total_minutes_played > 0, !is.na(starting_position)) |>
#   mutate(
#     extrapolated_minutes_played = minutes_played + prop * (total_minutes_played - starter_minutes_played)
#   ) |>
#   arrange(competition_id, season_id, player_id)

most_common_player_teams <- player_games |> 
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

players_season_games <- player_games |> 
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
    player_games |> 
      filter(is_starter) |> 
      count(competition_id, season_id, player_id, name = 'games_started'),
    by = join_by(competition_id, season_id, player_id)
  ) |> 
  left_join(
    most_common_player_teams |> 
      select(
        competition_id,
        season_id,
        player_id,
        team_id,
        team_name
      ),
    by = join_by(competition_id, season_id, player_id)
  ) |> 
  left_join(
    most_common_player_starting_positions |> 
      select(
        competition_id,
        season_id,
        player_id,
        starting_position_prop = prop,
        starting_position
      ),
    by = join_by(competition_id, season_id, player_id)
  ) |> 
  mutate(
    across(c(games_played, games_started), \(x) coalesce(x, 0L))
  )

vaep_by_player_season <- all_vaep |> 
  # filter(season_id == 2023L) |> 
  group_by(in_test, competition_id, season_id, player_id) |> 
  summarize(
    n_actions = sum(!is.na(action_id)),
    n_actions_atomic = n(),
    across(c(g, matches('vaep|xt')), \(x) sum(x, na.rm = TRUE))
  ) |> 
  ungroup() |> 
  inner_join(
    players_season_games,
    by = join_by(competition_id, season_id, player_id)
  ) |> 
  mutate(
    across(c(g, matches('vaep|xt')), list(p90 = \(x) x * 90 / minutes_played))
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
    starting_position,
    starting_position_prop,
    n_actions,
    n_actions_atomic,
    minutes_played,
    games_played,
    games_started,
    g,
    xt,
    ovaep,
    dvaep,
    vaep,
    ovaep_atomic,
    dvaep_atomic,
    vaep_atomic,
    xt_p90,
    ovaep_p90,
    dvaep_p90,
    vaep_p90,
    ovaep_atomic_p90,
    dvaep_atomic_p90,
    vaep_atomic_p90
  ) |> 
  arrange(desc(vaep_atomic))
export_parquet(players_season_games)
export_parquet(vaep_by_player_season)

## debug ----
# vaep_by_player_season |> arrange(desc(vaep_atomic))
# vaep_by_player_season |>
#   group_by(in_test, competition_id, season_id) |>
#   summarize(
#     across(
#       c(
#         g,
#         xt,
#         ovaep,
#         vaep,
#         ovaep_atomic,
#         vaep_atomic,
#       ),
#       \(x) sum(x, na.rm = TRUE)
#     )
#   ) |>
#   ungroup() |>
#   arrange(competition_id, season_id)

vaep_by_player_season |> 
  select(-in_test) |> 
  group_split(season_id) |> 
  walk(
    ~{
      path <- file.path('data/final/shared/', .x$season_id[1], 'vaep_by_player_season.csv')
      dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
      readr::write_csv(.x,  path, na = '')
    }
  )
players_season_games |> 
  group_split(season_id) |> 
  walk(
    ~{
      path <- file.path('data/final/shared/', .x$season_id[1], 'players_season_games.csv')
      dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
      readr::write_csv(.x,  path, na = '')
    }
  )

all_vaep |> 
  group_split(season_id) |> 
  walk(
    ~{
      path <- file.path('data/final/shared/', .x$season_id[1], 'events_with_vaep.parquet')
      dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
      arrow::write_parquet(.x,  path)
    }
  )
