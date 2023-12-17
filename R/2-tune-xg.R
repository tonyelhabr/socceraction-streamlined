library(dplyr)
library(lubridate)

library(tidymodels)
library(finetune)
library(bonsai)
library(themis)

library(vip)
library(pdp)
library(ggplot2)
library(scales)

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
    scores = factor(scores, levels = c('yes', 'no')),
    game_id,
    team_id,
    opponent_team_id = ifelse(team_id == home_team_id, away_team_id, home_team_id),
    elo = ifelse(team_id == home_team_id, home_elo, away_elo),
    opponent_elo = ifelse(team_id == home_elo, away_elo, home_elo),
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

init_split <- split_train_test(open_play_shots, games = games)
split <- make_splits(init_split$train, init_split$test)
train <- training(split)
test <- testing(split)

set.seed(42)
train_folds <- vfold_cv(train, strata = scores, v = 5)

rec_base <- recipe(
  scores ~ 
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

rec_elo <- recipe(
  scores ~ 
    elo +
    opponent_elo +
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


## https://jlaw.netlify.app/2022/01/24/predicting-when-kickers-get-iced-with-tidymodels/
## https://juliasilge.com/blog/childcare-costs/
spec <- boost_tree(
  # trees = tune(),
  # learn_rate = tune(),
  trees = 500,
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
  size = 50
)

wf_sets <- workflow_set(
  preproc = list(
    base = rec_base, 
    elo = rec_elo
  ),
  models = list(model = spec),
  cross = TRUE
)

met_set <- metric_set(f_meas, accuracy, roc_auc, sensitivity)
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

perf_stats |> 
  ggplot() +
  aes(x = wflow_id, color = wflow_id, y = mean) +
  geom_pointrange(
    aes(
      y = mean, 
      ymin = mean - 1.96*std_err, 
      ymax = mean + 1.96*std_err
    )
  ) + 
  facet_wrap(~.metric, scales = 'free_y') + 
  guides(color = 'none') +
  labs(
    title = 'Performance Metric for Tuned Results',
    x = 'Model Spec',
    y = 'Metric Value',
    color = 'Model Config'
  )

tuned_results |> 
  rank_results(rank_metric = 'f_meas') |>
  select(wflow_id, .config, .metric, mean, std_err) |>
  filter(.metric == 'f_meas') 

best_base_set <- tuned_results |>
  extract_workflow_set_result('base_model') |> 
  select_best(metric = 'f_meas')
best_base_set

best_elo_set <- tuned_results |>
  extract_workflow_set_result('elo_model') |> 
  select_best(metric = 'f_meas')
best_elo_set

# tuned_results |>
#   extract_workflow_set_result('base_model') |> 
#   plot_race()
# 
# tuned_results |>
#   extract_workflow_set_result('elo_model') |> 
#   plot_race()

final_base_fit <- tuned_results |>
  extract_workflow('base_model') |>
  finalize_workflow(best_base_set) |> 
  last_fit(
    split,
    metrics = met_set
  )

final_elo_fit <- tuned_results |>
  extract_workflow('elo_model') |>
  finalize_workflow(best_elo_set) |> 
  last_fit(
    split,
    metrics = met_set
  )

collect_metrics(final_base_fit)
collect_metrics(final_elo_fit)

final_base_fit |> 
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

final_base_fit |> 
  collect_predictions() |>
  conf_mat(scores, .pred_class) |>
  autoplot(type = 'heatmap')

library(vip)

final_base_fit |> 
  extract_fit_parsnip() |>
  vip(geom = 'point', include_type = TRUE) + 
  geom_text(
    aes(label = scales::percent(Importance, accuracy = 1)),
    nudge_y = 0.02
  )

library(vip)

final_elo_fit |> 
  extract_fit_parsnip() |>
  vip(geom = 'point', include_type = TRUE) + 
  geom_text(
    aes(label = scales::percent(Importance, accuracy = 1)),
    nudge_y = 0.02
  )

library(pdp)

##Get Processed Training Data
model_object <- extract_fit_engine(final_fit)

fitted_data <- rec_smote %>%
  prep() %>%
  bake(new_data = model_data) %>%
  select(-is_iced)
