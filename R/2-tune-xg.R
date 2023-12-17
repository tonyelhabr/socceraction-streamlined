library(dplyr)
library(lubridate)
library(tidymodels)
library(finetune)
# library(lightgbm)
library(bonsai)
library(themis)

source(file.path('R', 'helpers.R'))

games <- import_parquet('games')
elo <- import_parquet('clubelo-ratings')
xy <- import_xy(games = games, drop_action_regex = '_a2$')

open_play_shots <- xy |> 
  dplyr::filter(type_shot_a0 == 1) |> 
  dplyr::left_join(
    games |> 
      dplyr::transmute(
        game_id, 
        date = lubridate::date(game_date),
        home_team_id,
        away_team_id
      ),
    by = dplyr::join_by(game_id)
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
    scores,
    game_id,
    team_id,
    opponent_team_id = ifelse(team_id == home_team_id, away_team_id, home_team_id),
    elo = ifelse(team_id == home_team_id, home_elo, away_elo),
    opponent_elo = ifelse(team_id == home_elo, away_elo, home_elo),
    start_x_a0,
    start_y_a0,
    start_dist_to_goal_a0,
    start_angle_to_goal_a0,
    # type_pass_a1,
    # type_cross_a1,
    # type_dribble_a1,
    # type_shot_a1,
    bodypart_foot_a0,
    bodypart_head_a0,
    bodypart_other_a0
  )

split <- split_train_test(open_play_shots, games = games)
train <- split$train
test <- split$test

set.seed(42)
train_folds <- vfold_cv(train, strata = scores, v = 5)

rec_base <- recipe(
  scores ~ 
    start_x_a0 +
    start_y_a0 +
    start_dist_to_goal_a0 +
    start_angle_to_goal_a0 +
    # type_pass_a1 +
    # type_cross_a1 +
    # type_dribble_a1 +
    # type_shot_a1 +
    bodypart_foot_a0 +
    bodypart_head_a0 +
    bodypart_other_a0,
  data = train
)

rec_elo <- recipe(
  scores ~ 
    elo +
    opponent_elo +
    start_x_a0 +
    start_y_a0 +
    start_dist_to_goal_a0 +
    start_angle_to_goal_a0 +
    # type_pass_a1 +
    # type_cross_a1 +
    # type_dribble_a1 +
    # type_shot_a1 +
    bodypart_foot_a0 +
    bodypart_head_a0 +
    bodypart_other_a0,
  data = train
)


rec_smote <- rec_base |> 
  step_smote(all_outcomes())


## https://jlaw.netlify.app/2022/01/24/predicting-when-kickers-get-iced-with-tidymodels/
## https://juliasilge.com/blog/childcare-costs/
spec <- boost_tree(
  # trees = tune(),
  # learn_rate = tune(),
  trees = 300,
  learn_rate = 0.01,
  tree_depth = tune(),
  min_n = tune(), 
  loss_reduction = tune(),
  sample_size = tune(), 
  mtry = tune(),
  stop_iter = tune()
) |>
  set_engine('xgboost') |> 
  set_mode('classification')

grid <- grid_latin_hypercube(
  # trees(),
  # learn_rate(),
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), train),
  stop_iter(range = c(10L, 50L)),
  size = 20
)

wf_sets <- workflow_set(
  preproc = list(
    base = rec_base, 
    elo = rec_elo
    # smote = rec_smote
  ),
  models = list(model = spec),
  cross = TRUE
)

met_set <- metric_set(f_meas, accuracy, roc_auc, mn_log_loss)
control <- control_race(
  save_pred = TRUE,
  parallel_over = 'everything',
  save_workflow = TRUE,
  verbose = TRUE,
  verbose_elim = TRUE
)

options(tidymodels.dark = TRUE)
tuned_results <- workflow_map(
  wf_sets,
  fn = 'tune_race_anova',
  grid = grid,
  control = control,
  metrics = met_set,
  resamples = train_folds,
  seed = 42
)

autoplot(tuned_results)

perf_stats <- map_dfr(
  c('f_meas', 'accuracy', 'roc_auc', 'mn_log_loss'),
  \(.x) {
    rank_results(tuned_results, rank_metric = .x, select_best = TRUE) |> 
      filter(.metric == .x)
  }
)

rank_results(tuned_results, rank_metric = 'f_meas') |>
  select(.config, .metric, mean, std_err) |>
  filter(.metric == 'f_meas')

best_set <- tuned_results |>
  extract_workflow_set_result('rec_model') %>% 
  select_best(metric = 'f_meas')
best_set

final_fit <- tuned_results %>%
  extract_workflow('rec_model') %>%
  finalize_workflow(best_set) 
