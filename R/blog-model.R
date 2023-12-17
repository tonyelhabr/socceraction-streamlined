library(dplyr)
library(lubridate)

library(tidymodels)

library(vip)
library(pdp)
library(ggplot2)
library(scales)
library(pdp)
library(furrr)
library(future)

source(file.path('R', 'helpers.R'))

read_parquet_from_url <- function(url) {
  load <- curl::curl_fetch_memory(url)
  arrow::read_parquet(load$content)
}

read_release <- function(name) {
  url <- sprintf('https://github.com/tonyelhabr/socceraction-streamlined/releases/download/data-processed/%s.parquet', name)
  read_parquet_from_url(url)
}

read_data <- function(name) {
  purrr::map_dfr(
    c(2013:2019, 2021:2022),
    \(season_start_year) {
      basename <- sprintf('8-%s-%s', season_start_year, name)
      cli::cli_inform(basename)
      read_release(basename)
    }
  )
}

x <- read_data('x')
y <- read_data('y')
games <- read_data('games')
elo <- read_parquet_from_url('https://github.com/tonyelhabr/socceraction-streamlined/raw/main/data/final/8/2013-2022/clubelo-ratings.parquet')

xy <- inner_join(
  y |> select(game_id, action_id, scores),
  x |> select(
    game_id,
    action_id,
    start_x_a0,
    start_y_a0,
    start_dist_to_goal_a0,
    start_angle_to_goal_a0,
    type_dribble_a1,
    type_pass_a1,
    type_cross_a1,
    type_corner_crossed_a1,
    type_shot_a1,
    type_freekick_crossed_a1,
    bodypart_foot_a0,
    bodypart_head_a0,
    bodypart_other_a0
  ),
  by = join_by(game_id, action_id)
) |> 
  inner_join(
    games |> select(competition_id, season_id, game_id),
    by = join_by(game_id)
  ) |> 
  mutate(
    scores = ifelse(scores, 'yes', 'no') |> factor(levels = c('yes', 'no')),
    across(where(is.logical), as.integer)
  )

