library(httr)
library(dplyr)

source(file.path('R', 'helpers.R'))

games <- import_parquet('games')

match_dates <- games |> 
  dplyr::distinct(date = lubridate::date(game_date)) |> 
  dplyr::mutate(
    prev_date = date - lubridate::days(1)
  )

RAW_DATA_DIR <- 'data/raw'
CLUBELO_RAW_DATA_DIR <- file.path(RAW_DATA_DIR, 'clubelo')
dir.create(CLUBELO_RAW_DATA_DIR, showWarnings = FALSE)

get_clubelo_ratings <- function(date) {
  path <- file.path(CLUBELO_RAW_DATA_DIR, paste0(date, '.rds'))
  if (file.exists(path)) {
    return(readRDS(path))
  }
  Sys.sleep(runif(1, 1, 3))
  message(sprintf('Scraping clubelo for %s.', date))
  url <- sprintf('http://api.clubelo.com/%s', date)
  resp <- httr::GET(url)
  httr::stop_for_status(resp)
  res <- httr::content(resp)
  saveRDS(res,path)
  invisible(res)
}

possibly_get_clubelo_ratings <- purrr::possibly(
  get_clubelo_ratings,
  otherwise = tibble::tibble(),
  quiet = FALSE
)

raw_clubelo_ratings <- purrr::map2_dfr(
  match_dates$prev_date,
  match_dates$date,
  \(prev_date, date) {
    possibly_get_clubelo_ratings(prev_date) |> 
      dplyr::mutate(
        date = .env$date,
        .before = 1
      )
  }
)

## created manually
clubelo_team_mapping <- readr::read_csv('data/manual/clubelo-team-mapping.csv') |> 
  ## West Brom had an abbreviation change in 2016
  ##   We don't actually need to respect that change since we can rely on the team_id being consistent
  dplyr::filter(team_opta != 'WBA')

raw_clubelo_ratings |> 
  dplyr::filter(
    ## don't filter to Tier 1 since that would mess with some promoted teams' first match of the season
    Country == 'ENG'
  ) |> 
  dplyr::inner_join(
    clubelo_team_mapping |> 
      dplyr::select(
        team_clubelo,
        team_opta,
        team_opta_id
      ),
    by = dplyr::join_by(Club == team_clubelo)
  ) |> 
  dplyr::distinct(
    date,
    team_id = team_opta_id,
    team = team_opta,
    elo = Elo
  ) |> 
  arrow::write_parquet(file.path(FINAL_DATA_DIR, 'clubelo-ratings.parquet'))
