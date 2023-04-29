library(tidyverse)
library(tidymodels)

c(
  'actions',
  'actions_atomic',
  'actiontypes',
  'actiontypes_atomic',
  'bodyparts',
  'games',
  'players',
  'results',
  'teams'
) |> 
  walk(
    ~{
      res <- readr::read_csv(file.path('test-data', sprintf('%s.csv', .x)), show_col_types = FALSE)
      assign(value = res, x = tools::file_path_sans_ext(basename(.x)), envir = .GlobalEnv)
    }
  )

av <- actions |>
  left_join(actiontypes) |>
  left_join(bodyparts) |>
  left_join(players) |>
  left_join(teams) |>
  left_join(results) |>
  left_join(games)

ava <- actions_atomic |>
  left_join(actiontypes_atomic) |>
  left_join(bodyparts) |>
  left_join(players) |>
  left_join(teams) |>
  left_join(games)

add_pos_cols <- function(d) {
  d <- df |> 
    left_join(
      actions |> select(game_id, team_id, period_id, action_id)
    )
  
  poss_changes <- d |> 
    arrange(game_id, action_id) |> 
    mutate(across(action_id, list(lag1 = lag))) |> 
    mutate(
      poss_change = case_when(
        # for some reason i'm having to create this column explicitly, instead of just doing lag(action_id)
        is.na(action_id_lag1) ~ TRUE, # for first record in data set
        team_id != lag(team_id) ~TRUE, 
        game_id != lag(game_id) ~ TRUE,
        period_id != lag(period_id) ~ TRUE,
        TRUE ~ FALSE
      ) 
    ) |> 
    select(-action_id_lag1) |> 
    relocate(poss_change, matches('lag1$'))

  poss_ids <- poss_changes |> 
    filter(poss_change) |> 
    group_by(game_id) |> 
    mutate(idx_inter = row_number()) |> 
    ungroup()
  
  suppressMessages(
    poss_changes |> 
      left_join(poss_ids |> select(game_id, action_id, poss_change, idx_inter)) |> 
      fill(idx_inter) |> 
      group_by(game_id, idx_inter) |> 
      mutate(
        idx_intra = row_number()
      ) |> 
      ungroup() |> 
      select(-poss_change) |> 
      relocate(matches('^idx'))
  )
}

av |> add_pos_cols()


player_games <- players |> 
  select(
    player_id,
    game_id,
    team_id,
    is_starter,
    starting_position,
    minutes_played
  ) |> 
  mutate(
    across(
      minutes_played, 
      ~case_when(
        game_id == 1485349L ~ ifelse(.x < 98L, .x, 98), 
        TRUE ~ .x)
    )
  )
av

ga
