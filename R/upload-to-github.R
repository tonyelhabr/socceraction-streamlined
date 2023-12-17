library(piggyback)

pb_new_release(
  tag = 'data-processed', 
  repo = 'tonyelhabr/socceraction-streamlined'
)

upload_to_github <- function(x) {
  
  paths <- dir_ls(
    file.path(PROCESSED_DATA_DIR, COMPETITION_ID, SEASON_END_YEARS),
    regexp = paste0('\\/', x, '\\.parquet$'),
    recurse = TRUE
  )
  
  tibble(path = paths) |> 
    mutate(
      name = gsub(paste0(PROCESSED_DATA_DIR, '/'), '', path),
      name = gsub('/', '-', name)
    ) |> 
    transpose() |> 
    unname() |> 
    walk(
      \(.x) {
        pb_upload(
          file = .x$path,
          name = .x$name,
          tag = 'data-processed'
        )
      }
    )
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
  walk(upload_to_github)

c(
  'x',
  'x_atomic',
  'y',
  'y_atomic'
) |> 
  walk(upload_to_github)

