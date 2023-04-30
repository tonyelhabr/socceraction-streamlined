library(dplyr)
library(purrr)
library(readr)
library(fs)
library(arrow)

COMPETITION_ID <- 8
SEASON_END_YEARS <- 2021:2023
c(
  'actions',
  'actions_atomic',
  'actiontypes',
  'actiontypes_atomic',
  'bodyparts',
  'games',
  'players',
  'results',
  'teams',
  'xt',
  'x',
  'x_atomic',
  'y',
  'y_atomic'
) |> 
  walk(
    ~{
      paths <- dir_ls(
        file.path('../whoscraped/data/processed', COMPETITION_ID, SEASON_END_YEARS),
        recurse = TRUE,
        regexp = paste0('\\/', .x, '\\.parquet$')
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

av <- actions |>
  left_join(
    actiontypes |> distinct(),
    by = join_by(type_id)
  ) |>
  left_join(
    bodyparts |> distinct(),
    by = join_by(bodypart_id)
  ) |>
  left_join(
    players,
    by = join_by(game_id, team_id, player_id)
  ) |>
  left_join(
    teams |> distinct(),
    by = join_by(team_id)
  ) |>
  left_join(
    results |> distinct(),
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
    actiontypes_atomic |> distinct(),
    by = join_by(type_id)
  ) |>
  left_join(
    bodyparts |> distinct(),
    by = join_by(bodypart_id)
  ) |>
  left_join(
    players,
    by = join_by(game_id, team_id, player_id)
  ) |>
  left_join(
    teams |> distinct(),
    by = join_by(team_id)
  ) |>
  left_join(
    games,
    by = join_by(game_id)
  )

export_parquet <- function(name, dir = 'data/final') {
  df <- get(name)
  write_parquet(
    df,
    file.path(dir, paste0(name, '.parquet'))
  )
}

c(
  'av',
  'ava',
  'games',
  'actions',
  'actions_atomic',
  'x',
  'x_atomic',
  'y',
  'y_atomic'
) |> 
  walk(export_parquet)
