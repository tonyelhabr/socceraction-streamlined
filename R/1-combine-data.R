library(dplyr)
library(purrr)

source(file.path('R', 'helpers.R'))

c(
  # 'actions',
  # 'actions_atomic',
  # 'actiontypes',
  # 'actiontypes_atomic',
  # 'bodyparts',
  # 'games',
  # 'players',
  # 'results',
  'teams',
  # 'xt',
  # 'x',
  # 'x_atomic',
  # 'y',
  # 'y_atomic'
  'players'
) |> 
  walk(
    ~{
      paths <- list.files(
        file.path(PROCESSED_DATA_DIR, COMPETITION_ID, SEASON_END_YEARS),
        pattern = paste0(.x, '\\.parquet$'),
        full.names = TRUE,
        recursive = TRUE
      )
      
      res <- map_dfr(
        paths,
        ~{
          competition_id <- as.integer(basename(dirname(dirname(.x))))
          season_end_year <- as.integer(basename(dirname(.x)))
          read_parquet(.x) |> 
            mutate(
              across(
                where(~any(class(.x) == 'integer64')),
                as.double
              )
            )
        }
      )
      assign(value = res, x = .x, envir = .GlobalEnv)
    }
  )

players <- players |> 
  mutate(
    across(
      minutes_played, 
      ~case_when(
        minutes_played > 1000 ~ ifelse(.x < 98L, .x, 98), 
        TRUE ~ .x)
    )
  )

actiontypes <- distinct(actiontypes)
actiontypes_atomic <- distinct(actiontypes_atomic)
bodyparts <- distinct(bodyparts)
teams <- distinct(teams)
results <- distinct(results)

av <- actions |>
  left_join(
    actiontypes,
    by = join_by(type_id)
  ) |>
  left_join(
    bodyparts,
    by = join_by(bodypart_id)
  ) |>
  left_join(
    players,
    by = join_by(game_id, team_id, player_id)
  ) |>
  left_join(
    teams,
    by = join_by(team_id)
  ) |>
  left_join(
    results,
    by = join_by(result_id)
  ) |>
  left_join(
    xt,
    by = join_by(game_id, action_id)
  ) |> 
  left_join(
    games,
    by = join_by(game_id)
  )

ava <- actions_atomic |>
  left_join(
    actiontypes_atomic,
    by = join_by(type_id)
  ) |>
  left_join(
    bodyparts,
    by = join_by(bodypart_id)
  ) |>
  left_join(
    players,
    by = join_by(game_id, team_id, player_id)
  ) |>
  left_join(
    teams,
    by = join_by(team_id)
  ) |>
  left_join(
    games,
    by = join_by(game_id)
  )

c(
  'av',
  'ava',
  'games',
  'players',
  'teams',
  'actions',
  'actions_atomic',
  'x',
  'x_atomic',
  'y',
  'y_atomic'
) |> 
  walk(
    ~{
      df <- get(name)
      export_parquet(df, name)
    }
  )
