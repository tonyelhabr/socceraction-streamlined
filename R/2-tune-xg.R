library(tidymodels)
library(finetune)
library(lightgbm)
library(bonsai)

select_xy <- function(df) {
  df |> 
    filter(bodypart_head/other_a0 != 1) |> 
    transmute(
      scores,
      start_x_a0,
      start_y_a0,
      start_dist_to_goal_a0,
      start_angle_to_goal_a0,
      bodypart_foot_a0,
      bodypart_head_a0,
      bodypart_other_a0
    )
}

set.seed(42)
tidy_split <- list()
tidy_split$train <- bind_cols(
  scores = split$train$scores,
  .select_xg_x(split$train)
)

tidy_split$test <- bind_cols(
  scores = split$test$scores,
  .select_xg_x(split$test)
)

train_folds <- vfold_cv(tidy_split$train, strata = scores, v = 5)

## https://jlaw.netlify.app/2022/01/24/predicting-when-kickers-get-iced-with-tidymodels/
## https://juliasilge.com/blog/childcare-costs/
spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  min_n = tune(), 
  loss_reduction = tune(),
  sample_size = tune(), 
  mtry = tune(),
  stop_iter = tune(),
  learn_rate = tune()
) |>
  set_engine('xgboost', validation = 0.2) |> 
  set_mode('classification')

rec <- recipe(
  scores ~ .,
  data = tidy_split$train
)

grid <- grid_latin_hypercube(
  trees(),
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), tidy_split$train),
  learn_rate(),
  stop_iter(range = c(10L, 50L)),
  size = 50
)

wf <- workflow() |> 
  add_model(spec) |> 
  add_recipe(rec)

met_set <- metric_set(f_meas, accuracy, roc_auc, mn_log_loss, brier_class)
control <- control_race(
  save_pred = TRUE,
  parallel_over = 'everything',
  save_workflow = TRUE
)

options(tidymodels.dark = TRUE)
tuned_results <- tune_race_anova(
  wf,
  grid = grid,
  control = control,
  metrics = met_set,
  resamples = train_folds
)

autoplot(tuned_results)

perf_stats <- map_dfr(
  c('f_meas', 'accuracy', 'roc_auc', 'mn_log_loss', 'brier_class'),
  \(.x) {
    rank_results(tuned_model, rank_metric = .x, select_best = TRUE) |> 
      filter(.metric == .x)
  }
)

rank_results(tuned_results, rank_metric = 'f_meas') |>
  select(.config, .metric, mean, std_err) |>
  filter(.metric == 'f_meas')
