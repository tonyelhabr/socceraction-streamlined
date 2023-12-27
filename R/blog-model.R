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

read_parquet_from_url <- function(url) {
  load <- curl::curl_fetch_memory(url)
  arrow::read_parquet(load$content)
}

REPO <- 'tonyelhabr/socceraction-streamlined'
read_socceraction_parquet_release <- function(name, tag) {
  url <- sprintf('https://github.com/%s/releases/download/%s/%s.parquet', REPO, tag, name)
  read_parquet_from_url(url)
}

read_socceraction_parquet_releases <- function(name, tag = 'data-processed') {
  purrr::map_dfr(
    setdiff(2013:2022, 2020),
    \(season_start_year) {
      basename <- sprintf('8-%s-%s', season_start_year, name)
      cli::cli_inform(basename)
      read_socceraction_parquet_release(basename, tag = tag)
    }
  )
}

read_socceraction_parquet <- function(name, branch = 'main') {
  url <- sprintf('https://github.com/%s/raw/%s/%s.parquet', REPO, branch, name)
  read_parquet_from_url(url)
}

x <- read_socceraction_parquet_releases('x')
y <- read_socceraction_parquet_releases('y')
actions <- read_socceraction_parquet_releases('actions')
players <- read_socceraction_parquet_releases('players')
games <- read_socceraction_parquet_releases('games')
team_elo <- read_socceraction_parquet('data/final/8/2013-2022/clubelo-ratings')

open_play_shots <- games |>
  dplyr::transmute(
    game_id,
    date = lubridate::date(game_date),
    home_team_id,
    away_team_id
  ) |> 
  dplyr::inner_join(
    x |> 
      dplyr::filter(type_shot_a0 == 1) |> 
      dplyr::select(
        game_id,
        action_id,

        ## features
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
      ) |> 
      dplyr::mutate(
        dplyr::across(-c(game_id, action_id), as.integer)
      ),
    by = dplyr::join_by(game_id),
    relationship = 'many-to-many'
  ) |> 
  dplyr::inner_join(
    y |> 
      dplyr::transmute(
        game_id, 
        action_id,
        scores = ifelse(scores, 'yes', 'no') |> factor(levels = c('yes', 'no'))
      ),
    by = dplyr::join_by(game_id, action_id)
  ) |> 
  dplyr::inner_join(
    actions |> 
      dplyr::select(
        game_id,
        action_id,
        team_id,
        player_id
      ),
    by = dplyr::join_by(game_id, action_id)
  ) |> 
  dplyr::inner_join(
    players |> 
      dplyr::select(
        game_id,
        team_id,
        player_id,
        player_name
      ),
    by = dplyr::join_by(game_id, team_id, player_id)
  ) |> 
  dplyr::left_join(
    elo |> dplyr::select(date, home_team_id = team_id, home_elo = elo),
    by = dplyr::join_by(date, home_team_id)
  ) |> 
  dplyr::left_join(
    elo |> dplyr::select(date, away_team_id = team_id, away_elo = elo),
    by = dplyr::join_by(date, away_team_id)
  ) |> 
  dplyr::transmute(
    date,
    game_id,
    team_id,
    action_id,
    player_id,
    player_name,
    
    scores,
    opponent_team_id = ifelse(team_id == home_team_id, away_team_id, home_team_id),
    elo = ifelse(team_id == home_team_id, home_elo, away_elo),
    opponent_elo = ifelse(team_id == home_team_id, away_elo, home_elo),
    elo_diff = elo - opponent_elo,
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
  )

open_play_shots |> 
  arrange(date, game_id, action_id, player_id) |> 
  group_by(player_id) |> 
  mutate(
    goals_in_prev10_shots = slider::slide_sum(scores == 'yes', before = 10L, after = 0L)
  ) |> 
  ungroup() |> 
  select(
    date, game_id, action_id, 
    player_id,
    goals_in_prev10_shots
  ) -> z
z |> arrange(desc(goals_in_prev10_shots))

train_game_ids <- games |> dplyr::filter(!(season_id %in% c(2021:2022))) |> dplyr::pull(game_id)
test_game_ids <- games |> dplyr::filter(season_id %in% c(2013:2019)) |> dplyr::pull(game_id)
train <- open_play_shots |> dplyr::filter(game_id %in% train_game_ids)
test <- open_play_shots |> dplyr::filter(game_id %in% test_game_ids)

init_split <- split_train_test(open_play_shots, games = games)
split <- rsample::make_splits(init_split$train, init_split$test)
train <- rsample::training(split)
test <- rsample::testing(split)

rec_elo <- recipes::recipe(
  scores ~ 
    elo +
    elo_diff +
    start_x_a0 +
    start_y_a0 +
    start_dist_to_goal_a0 +
    start_angle_to_goal_a0 +
    type_dribble_a1 +
    type_pass_a1 +
    type_cross_a1 +
    type_corner_crossed_a1 +
    type_shot_a1 +
    type_freekick_crossed_a1 +
    bodypart_foot_a0 +
    bodypart_head_a0 +
    bodypart_other_a0,
  data = train
)

rec_base <- rec_elo |> 
  recipes::step_rm(elo, elo_diff)

spec_base <- boost_tree(
  trees = 500,
  learn_rate = 0.01,
  tree_depth = 12,
  min_n = 20, 
  loss_reduction = 0.0009316,
  sample_size = 0.2373513,
  mtry = 11,
  stop_iter = 36
) |>
  set_engine('xgboost') |> 
  set_mode('classification')

spec_elo <- boost_tree(
  trees = 500,
  learn_rate = 0.01,
  tree_depth = 13,
  min_n = 31, 
  loss_reduction = 0.0006153,
  sample_size = 0.3222589,
  mtry = 12,
  stop_iter = 47
) |>
  set_engine('xgboost') |> 
  set_mode('classification')

wf_base <- workflow(
  preprocessor = rec_base,
  spec = spec_base
)

wf_elo <- workflow(
  preprocessor = rec_elo,
  spec = spec_elo
)

met_set <- metric_set(f_meas, accuracy, roc_auc, sensitivity)
fit_base <- last_fit(
  wf_base,
  split = split,
  metrics = met_set
)

collect_metrics(fit_base)

fit_base |> 
  collect_predictions() |>
  roc_curve(scores, .pred_yes) |>
  ggplot() +
  aes(
    x = 1 - specificity, 
    y = sensitivity
  ) +
  geom_abline(lty = 2, linewidth = 1.5) +
  geom_point() +
  coord_equal()

fit_base |> 
  extract_fit_parsnip() |>
  vip(geom = 'point', include_type = TRUE, num_features = 100) + 
  geom_text(
    aes(label = scales::percent(Importance, accuracy = 1)),
    nudge_y = 0.02
  )
