library(dplyr)
library(purrr)
library(fs)

source(file.path('R', 'helpers.R'))

do_import_parquets <- function(x, assign = TRUE) {
  message(sprintf('Processing %s.', x))
  paths <- dir_ls(
    file.path(PROCESSED_DATA_DIR, COMPETITION_ID, SEASON_END_YEARS),
    regexp = paste0('\\/', x, '\\.parquet$'),
    recurse = TRUE
  )
  message(sprintf('Found %s file paths.', length(paths)))
  
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
  
  if (isFALSE(assign)) {
    return(res)
  }
  
  assign(value = res, x = x, envir = .GlobalEnv)
}

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
  'xt'
) |> 
  walk(do_import_parquets)

players <- players |> 
  mutate(
    across(
      minutes_played, 
      ~case_when(
        minutes_played > 1000 ~ ifelse(.x < 98L, .x, 98), 
        TRUE ~ .x)
    )
  )

## Same across all seasons
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
  'actions',
  'actions_atomic',
  'games',
  'players',
  'teams'
) |> 
  walk(
    ~{
      df <- get(.x)
      export_parquet(df, .x)
    }
  )

c(
  'x',
  'x_atomic',
  'y',
  'y_atomic'
) |> 
  walk(
    ~{
      res <- do_import_parquets(.x, assign = FALSE)
      export_parquet(res, .x)
    }
  )
