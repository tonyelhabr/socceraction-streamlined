library(dplyr)
library(purrr)
library(yardstick)

source(file.path('R', 'helpers.R'))

# mse <- function(data, ...) {
#   UseMethod('mse')
# }
# 
# mse <- yardstick::new_prob_metric(
#   mse, 
#   direction = 'minimize'
# )
# 
# mse_vec <- function(truth, estimate, na_rm = TRUE, ...) {
#   mse_impl <- function(truth, estimate) {
#     mean((truth - estimate)^2)
#   }
#   
#   yardstick::metric_vec_template(
#     metric_impl = mse_impl,
#     truth = truth,
#     estimate = estimate,
#     na_rm = na_rm,
#     cls = 'prob',
#     ...
#   )
# }
# 
# mse.data.frame <- function(data, truth, estimate, na_rm = TRUE, ...) {
#   yardstick::prob_metric_summarizer(
#     metric_nm = 'mse',
#     metric_fn = mse_vec,
#     data = data,
#     truth = !!enquo(truth),
#     estimate = !!enquo(estimate),
#     na_rm = na_rm,
#     ...
#   )
# }

c(
  'preds',
  'preds_atomic',
) |> 
  walk(
    ~{
      res <- import_parquet(.x)
      assign(value = res, x = .x, envir = .GlobalEnv)
    }
  )

met_set <- metric_set(
  roc_auc,
  mn_log_loss,
  brier_class(event_level = 'second')
)

mean(((1L - as.integer(preds$scores)) - preds$ovaep)^2)
preds |> 
  met_set(
    truth = scores,
    ovaep,
    event_level = 'second'
  )

preds |> 
  met_set(
    truth = concedes,
    dvaep,
    event_level = 'second'
  )

preds_atomic |> 
  met_set(
    truth = scores,
    ovaep_atomic,
    event_level = 'second'
  )

preds_atomic |> 
  met_set(
    truth = concedes,
    dvaep_atomic,
    event_level = 'second'
  )

