#%%
import os
import pandas as pd
import xgboost
import functools
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss

#%%
def create_dir(dir):
  if not os.path.exists(dir):
    os.makedirs(dir)
  return dir

def path_exists(path):
  if not os.path.exists(path):
    return False
  else:
    return True

def generate_path(basename, ext, dir):
  create_dir(dir)
  return os.path.join(dir, f'{str(basename)}.{ext}')

def generate_parquet_path(basename, dir='../data/final/'):
  return generate_path(dir=dir, ext='parquet', basename=basename)

def do_if_parquet_path_not_exists(path, overwrite=True):
  def decorator(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
      if path_exists(path) and not overwrite:
        # print(f'Reading from {path}.')
        return pd.read_parquet(path)

      df = f(*args, **kwargs)
      df.to_parquet(path, index=False)
      return df
    return wrapper
  return decorator

#%%
x_trn = (
    pd.concat([
    pd.read_parquet(os.path.join('../data/processed/8/2020/x.parquet')),
    pd.read_parquet(os.path.join('../data/processed/8/2021/x.parquet')),
    pd.read_parquet(os.path.join('../data/processed/8/2022/x.parquet'))
  ])
  .drop(['game_id', 'action_id'], axis=1)
)

y_trn = (
  pd.concat([
    pd.read_parquet(os.path.join('../data/processed/8/2020/y.parquet')),
    pd.read_parquet(os.path.join('../data/processed/8/2021/y.parquet')),
    pd.read_parquet(os.path.join('../data/processed/8/2022/y.parquet'))
  ])
  .drop(['game_id', 'action_id'], axis=1)
)

x_tst = (
  pd.read_parquet(os.path.join('../data/processed/8/2023/x.parquet'))
  .drop(['game_id', 'action_id'], axis=1)
)
y_tst = (
  pd.read_parquet(os.path.join('../data/processed/8/2023/y.parquet'))
  .drop(['game_id', 'action_id'], axis=1)
)

#%%
x_atomic_trn = (
    pd.concat([
    pd.read_parquet(os.path.join('../data/processed/8/2020/x_atomic.parquet')),
    pd.read_parquet(os.path.join('../data/processed/8/2021/x_atomic.parquet')),
    pd.read_parquet(os.path.join('../data/processed/8/2022/x_atomic.parquet'))
  ])
  .drop(['game_id', 'action_id'], axis=1)
)

y_atomic_trn = (
  pd.concat([
    pd.read_parquet(os.path.join('../data/processed/8/2020/y_atomic.parquet')),
    pd.read_parquet(os.path.join('../data/processed/8/2021/y_atomic.parquet')),
    pd.read_parquet(os.path.join('../data/processed/8/2022/y_atomic.parquet'))
  ])
  .drop(['game_id', 'action_id'], axis=1)
)

x_atomic_tst = (
  pd.read_parquet(os.path.join('../data/processed/8/2023/x_atomic.parquet'))
  .drop(['game_id', 'action_id'], axis=1)
)
y_atomic_tst = (
  pd.read_parquet(os.path.join('../data/processed/8/2023/y_atomic.parquet'))
  .drop(['game_id', 'action_id'], axis=1)
)

#%%
actions = (
  pd.concat([
    pd.read_parquet(os.path.join('../data/processed/8/2020/actions.parquet')),
    pd.read_parquet(os.path.join('../data/processed/8/2021/actions.parquet')),
    pd.read_parquet(os.path.join('../data/processed/8/2022/actions.parquet')),
    pd.read_parquet(os.path.join('../data/processed/8/2023/actions.parquet'))
  ])
)

actions_atomic = (
  pd.concat([
    pd.read_parquet(os.path.join('../data/processed/8/2020/actions_atomic.parquet')),
    pd.read_parquet(os.path.join('../data/processed/8/2021/actions_atomic.parquet')),
    pd.read_parquet(os.path.join('../data/processed/8/2022/actions_atomic.parquet')),
    pd.read_parquet(os.path.join('../data/processed/8/2023/actions_atomic.parquet'))
  ])
)

#%%
models = {}
for col in ['scores', 'concedes']:
  print(f'Fitting model for {col}.')
  model = xgboost.XGBClassifier(n_estimators=100, max_depth=3, n_jobs=-3, verbosity=1)
  model.fit(x_trn, y_trn[col])
  model.save_model(f'../data/final/model_{col}.model')
  models[col] = model

#%%
models_atomic = {}
for col in ['scores', 'concedes']:
  print(f'Fitting model for {col}.')
  model = xgboost.XGBClassifier(n_estimators=100, max_depth=3, n_jobs=-3, verbosity=1)
  model.fit(x_atomic_trn, y_atomic_trn[col])
  model.save_model(f'../data/final/model_{col}_atomic.model')
  models_atomic[col] = model

#%%
def evaluate(y, y_hat):
  p = sum(y) / len(y)
  base = [p] * len(y)
  brier = brier_score_loss(y, y_hat)
  print(f"  Brier score: %.5f (%.5f)" % (brier, brier / brier_score_loss(y, base)))
  ll = log_loss(y, y_hat)
  print(f"  log loss score: %.5f (%.5f)" % (ll, ll / log_loss(y, base)))
  print(f"  ROC AUC: %.5f" % roc_auc_score(y, y_hat))

def predict_and_evaluate_set(x, y, models):
  preds = pd.DataFrame()
  for col in ['scores', 'concedes']:
    preds[col] = [p[1] for p in models[col].predict_proba(x)]
    print(f"### Y: {col} ###")
    evaluate(y[col], preds[col])
  
  return(preds)

@do_if_parquet_path_not_exists(path=generate_parquet_path('preds_train'))
def predict_and_evaluate_train_set(x, y, models):
  return(predict_and_evaluate_set(x, y, models))


@do_if_parquet_path_not_exists(path=generate_parquet_path('preds_test'))
def predict_and_evaluate_test_set(x, y, models):
  return(predict_and_evaluate_set(x, y, models))

@do_if_parquet_path_not_exists(path=generate_parquet_path('preds_atomic_train'))
def predict_and_evaluate_atomic_train_set(x, y, models):
  return(predict_and_evaluate_set(x, y, models))


@do_if_parquet_path_not_exists(path=generate_parquet_path('preds_atomic_test'))
def predict_and_evaluate_atomic_test_set(x, y, models):
  return(predict_and_evaluate_set(x, y, models))

#%%
preds_trn = predict_and_evaluate_train_set(x_trn, y_trn, models)
preds_tst = predict_and_evaluate_test_set(x_tst, y_tst, models)

#%%
preds_atomic_trn = predict_and_evaluate_train_set(x_atomic_trn, y_atomic_trn, models_atomic)
preds_atomic_tst = predict_and_evaluate_test_set(x_atomic_tst, y_atomic_tst, models_atomic)

#%%
import socceraction.vaep.formula as vaepformula
import socceraction.atomic.vaep.formula as vaepformula_atomic

#%%
preds_trn.concedes.head()

#%%
values_trn = vaepformula(actions_trn, preds_trn.scores, preds_tst.concedes)
values_tst = vaepformula_atomic(actions_tst, preds_trn.scores, preds_tst.concedes)