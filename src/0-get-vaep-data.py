#%%
## use pip (pip install -r requirements.txt)
import os
import pickle
import pandas as pd
from socceraction.spadl.opta import OptaLoader
import socceraction.spadl.opta as converter
import socceraction.spadl as spadl
import socceraction.atomic.spadl as atomicspadl
import tqdm
import functools

import socceraction.vaep.features as fs
import socceraction.vaep.labels as lab
import socceraction.atomic.vaep.features as fs_atomic
import socceraction.atomic.vaep.labels as lab_atomic
import socceraction.xthreat as xthreat

#%%
## globals
COMPETITION_ID = 85
SEASON_ID = 2013
## 914834, 914899, 915139 in MLS 2015 have some issue with type_name
RAW_DATA_DIR = f'../data/raw/'
PROCESSED_DATA_DIR = f'../data/processed/{COMPETITION_ID}/{SEASON_ID}'

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

def generate_parquet_path(basename, dir=PROCESSED_DATA_DIR):
  return generate_path(dir=dir, ext='parquet', basename=basename)

def generate_pickle_path(basename, dir=PROCESSED_DATA_DIR):
  return generate_path(dir=dir, ext='pickle', basename=basename)

def do_if_parquet_path_not_exists(path, overwrite=False):
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

def read_pickle(path):
  with open(path, 'rb') as handle:
    res = pickle.load(handle)
  return res

def write_pickle(data, path):
  with open(path, 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
  return path

def do_if_pickle_path_not_exists(path, overwrite=False):
  def decorator(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
      if path_exists(path) and not overwrite:
        # print(f'Reading from {path}.')
        return read_pickle(path)

      data = f(*args, **kwargs)
      write_pickle(data=data, path=path)
      return data
    return wrapper
  return decorator

def possibly(default_value):
  def decorator(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
      try:
        return f(*args, **kwargs)
      except Exception:
        return default_value
    return wrapper
  return decorator

@do_if_parquet_path_not_exists(path=generate_parquet_path('games'))
def get_games(loader):
  games = list(
    loader.games(row.competition_id, row.season_id)
    for row in selected_competitions.itertuples()
  )

  return pd.concat(games, sort=True).reset_index(drop=True)

def get_game_teams(loader, game_id):
  @do_if_parquet_path_not_exists(path=generate_parquet_path(game_id, dir=os.path.join(PROCESSED_DATA_DIR, 'teams')))
  def f(game_id):
    return loader.teams(game_id)
    
  return f(game_id)

@do_if_parquet_path_not_exists(path=generate_parquet_path('teams'))
def get_teams(loader, games):
  res = []
  for game in tqdm.tqdm(list(games.itertuples()), desc='Loading teams by game'):
    res.append(get_game_teams(loader, game.game_id))
  
  return pd.concat(res).drop_duplicates('team_id').reset_index(drop=True)

def get_game_players(loader, game_id):
  @do_if_parquet_path_not_exists(path=generate_parquet_path(game_id, dir=os.path.join(PROCESSED_DATA_DIR, 'players')))
  def f(game_id):
    return loader.players(game_id)
    
  return f(game_id)

@do_if_parquet_path_not_exists(path=generate_parquet_path('players'))
def get_players(loader, games):
  res = []
  for game in tqdm.tqdm(list(games.itertuples()), desc='Loading players by game'):
    res.append(get_game_players(loader, game.game_id))
  
  return pd.concat(res).reset_index(drop=True)

@possibly(pd.DataFrame())
def possibly_load_events(loader, game_id):
  return loader.events(game_id)

def get_game_actions(loader, game_id, home_team_id):
  @do_if_parquet_path_not_exists(path=generate_parquet_path(game_id, dir=os.path.join(PROCESSED_DATA_DIR, 'actions')))
  @possibly(pd.DataFrame())
  def f(events, home_team_id):
    return converter.convert_to_actions(events, home_team_id)
  
  return f(possibly_load_events(loader, game_id), home_team_id)
  
@do_if_parquet_path_not_exists(path=generate_parquet_path('actions'))
def get_actions(loader, games):
  res = []
  for game in tqdm.tqdm(list(games.itertuples()), desc='Loading actions by game'):
    res.append(get_game_actions(loader, game_id=game.game_id, home_team_id=game.home_team_id))
  
  return pd.concat(res).reset_index(drop=True)

def get_game_actions_atomic(loader, game_id, home_team_id):
  @do_if_parquet_path_not_exists(path=generate_parquet_path(game_id, dir=os.path.join(PROCESSED_DATA_DIR, 'actions_atomic')))
  @possibly(pd.DataFrame())
  def f(loader, game_id, home_team_id):
    actions = get_game_actions(loader, game_id=game_id, home_team_id=home_team_id)
    return atomicspadl.convert_to_atomic(actions)
  
  return f(loader, game_id, home_team_id)

@do_if_parquet_path_not_exists(path=generate_parquet_path('actions_atomic'))
def get_actions_atomic(loader, games):
  res = []
  for game in tqdm.tqdm(list(games.itertuples()), desc='Loading atomic actions by game'):
    res.append(get_game_actions_atomic(loader, game_id=game.game_id, home_team_id=game.home_team_id))
  
  return pd.concat(res).reset_index(drop=True)

def _get_game_id(actions):
  return actions[['game_id']]

def _get_action_id(actions):
  return actions[['action_id']]

@do_if_parquet_path_not_exists(path=generate_parquet_path('bodyparts'))
def get_bodyparts():
  return spadl.bodyparts_df()

@do_if_parquet_path_not_exists(path=generate_parquet_path('results'))
def get_results():
  return spadl.results_df()

@do_if_parquet_path_not_exists(path=generate_parquet_path('actiontypes'))
def get_actiontypes():
  return spadl.actiontypes_df()

@do_if_parquet_path_not_exists(path=generate_parquet_path('actiontypes_atomic'))
def get_actiontypes_atomic():
  return atomicspadl.actiontypes_df()

_xfns = [
  fs.actiontype_onehot,
  fs.bodypart_onehot,
  fs.result_onehot,
  fs.goalscore,
  fs.startlocation,
  fs.endlocation,
  fs.movement,
  fs.space_delta,
  fs.startpolar,
  fs.endpolar,
  fs.team,
  fs.time,
  fs.time_delta
]
## model only needs 1, but get 2 for xg model
_n_prev_actions = 2

def get_game_gamestates(loader, game_id, home_team_id): 
  @do_if_pickle_path_not_exists(path=generate_pickle_path(game_id, dir=os.path.join(PROCESSED_DATA_DIR, 'gamestates')), overwrite = True)
  @possibly(pd.DataFrame())
  def f(loader, game_id, home_team_id):
    actions = get_game_actions(loader=loader, game_id=game_id, home_team_id=home_team_id)
    gamestates = fs.gamestates(spadl.add_names(actions), _n_prev_actions)
    return fs.play_left_to_right(gamestates, home_team_id)
  
  return f(loader, game_id, home_team_id)

def get_game_gamestate_actions(loader, game_id, home_team_id):
  @do_if_parquet_path_not_exists(path=generate_parquet_path(game_id, dir=os.path.join(PROCESSED_DATA_DIR, 'gamestate_actions')), overwrite = True)
  @possibly(pd.DataFrame())
  def f(loader, game_id, home_team_id):
    gamestates = get_game_gamestates(loader=loader, game_id=game_id, home_team_id=home_team_id)
    return pd.DataFrame(gamestates[0])
  
  return f(loader, game_id, home_team_id)

@do_if_parquet_path_not_exists(path=generate_parquet_path('gamestate_actions'), overwrite = True)
def get_gamestate_actions(loader, games):
  res = []
  for game in tqdm.tqdm(list(games.itertuples()), desc='Loading gamestate actions by game'):
    res.append(get_game_gamestate_actions(loader=loader, game_id=game.game_id, home_team_id=game.home_team_id))
  
  return pd.concat(res).reset_index(drop=True)

def get_game_x(loader, game_id, home_team_id):
  @do_if_parquet_path_not_exists(path=generate_parquet_path(game_id, dir=os.path.join(PROCESSED_DATA_DIR, 'x')), overwrite = True)
  def f(loader, game_id, home_team_id):
    gamestates = get_game_gamestates(loader=loader, game_id=game_id, home_team_id=home_team_id)
    x = pd.concat([fn(gamestates) for fn in _xfns], axis=1)
    return pd.concat([_get_game_id(gamestates[0]), _get_action_id(gamestates[0]), x], axis=1)
  
  return f(loader, game_id, home_team_id)

@do_if_parquet_path_not_exists(path=generate_parquet_path('x'), overwrite = True)
def get_x(loader, games):
  res = []
  for game in tqdm.tqdm(list(games.itertuples()), desc='Loading x by game'):
    res.append(get_game_x(loader=loader, game_id=game.game_id, home_team_id=game.home_team_id))
  
  return pd.concat(res).reset_index(drop=True)

_xfns_atomic = [
  fs_atomic.actiontype_onehot,
  fs_atomic.bodypart_onehot,
  fs_atomic.goalscore,
  fs_atomic.location,
  fs_atomic.polar,
  fs_atomic.direction,
  fs_atomic.team,
  fs_atomic.time,
  fs_atomic.time_delta
]

def get_game_gamestates_atomic(loader, game_id, home_team_id): 
  @do_if_pickle_path_not_exists(path=generate_pickle_path(game_id, dir=os.path.join(PROCESSED_DATA_DIR, 'gamestates_atomic')))
  def f(loader, game_id, home_team_id):
    actions_atomic = get_game_actions_atomic(loader=loader, game_id=game_id, home_team_id=home_team_id)
    gamestates_atomic = fs_atomic.gamestates(atomicspadl.add_names(actions_atomic), _n_prev_actions)
    return fs_atomic.play_left_to_right(gamestates_atomic, home_team_id)
  
  return f(loader, game_id, home_team_id)

def get_game_x_atomic(loader, game_id, home_team_id):
  @do_if_parquet_path_not_exists(path=generate_parquet_path(game_id, dir=os.path.join(PROCESSED_DATA_DIR, 'x_atomic')))
  def f(loader, game_id, home_team_id):
    gamestates_atomic = get_game_gamestates_atomic(loader=loader, game_id=game_id, home_team_id=home_team_id)
    x_atomic = pd.concat([fn(gamestates_atomic) for fn in _xfns_atomic], axis=1)
    return pd.concat([_get_game_id(gamestates_atomic[0]), _get_action_id(gamestates_atomic[0]), x_atomic], axis=1)
  
  return f(loader, game_id, home_team_id)

@do_if_parquet_path_not_exists(path=generate_parquet_path('x_atomic'), overwrite = True)
def get_x_atomic(loader, games):
  res = []
  for game in tqdm.tqdm(list(games.itertuples()), desc='Loading x atomic by game'):
    res.append(get_game_x_atomic(loader=loader, game_id=game.game_id, home_team_id=game.home_team_id))
  
  return pd.concat(res).reset_index(drop=True)

_yfns = [
  lab.scores, 
  lab.concedes,
  lab.goal_from_shot
]
def get_game_y(loader, game_id, home_team_id):
  @do_if_parquet_path_not_exists(path=generate_parquet_path(game_id, dir=os.path.join(PROCESSED_DATA_DIR, 'y')))
  def f(loader, game_id, home_team_id):
    actions = get_game_actions(loader=loader, game_id=game_id, home_team_id=home_team_id)
    y = pd.concat([fn(spadl.add_names(actions)) for fn in _yfns], axis=1)
    return pd.concat([_get_game_id(actions), _get_action_id(actions), y], axis=1)
  
  return f(loader, game_id, home_team_id)

@do_if_parquet_path_not_exists(path=generate_parquet_path('y'))
def get_y(loader, games):
  res = []
  for game in tqdm.tqdm(list(games.itertuples()), desc='Loading y by game'):
    res.append(get_game_y(loader=loader, game_id=game.game_id, home_team_id=game.home_team_id))
  
  return pd.concat(res).reset_index(drop=True)

_yfns_atomic = [
  lab_atomic.scores, 
  lab_atomic.concedes,
  lab_atomic.goal_from_shot
]
def get_game_y_atomic(loader, game_id, home_team_id):
  @do_if_parquet_path_not_exists(path=generate_parquet_path(game_id, dir=os.path.join(PROCESSED_DATA_DIR, 'y_atomic')))
  def f(loader, game_id, home_team_id):
    actions_atomic = get_game_actions_atomic(loader=loader, game_id=game_id, home_team_id=home_team_id)
    y_atomic = pd.concat([fn(atomicspadl.add_names(actions_atomic)) for fn in _yfns_atomic], axis=1)
    return pd.concat([_get_game_id(actions_atomic), _get_action_id(actions_atomic), y_atomic], axis=1)
  
  return f(loader, game_id, home_team_id)

@do_if_parquet_path_not_exists(path=generate_parquet_path('y_atomic'))
def get_y_atomic(loader, games):
  res = []
  for game in tqdm.tqdm(list(games.itertuples()), desc='Loading y atomic by game'):
    res.append(get_game_y_atomic(loader=loader, game_id=game.game_id, home_team_id=game.home_team_id))
  
  return pd.concat(res).reset_index(drop=True)

def load_model(path_or_buf: str):
    """Create a model from a pre-computed xT value surface.
    The value surface should be provided as a JSON file containing a 2D
    matrix. Karun Singh provides such a grid at the follwing url:
    https://karun.in/blog/data/open_xt_12x8_v1.json
    Parameters
    ----------
    path_or_buf : a valid JSON str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file.
    Returns
    -------
    ExpectedThreat
        An xT model that uses the given value surface to value actions.
    """
    grid = pd.read_json(path_or_buf)
    model = xthreat.ExpectedThreat()
    model.xT = grid.values
    model.w, model.l = model.xT.shape
    return model

@do_if_parquet_path_not_exists(path=generate_parquet_path('xt'))
def get_xt(gamestate_actions):
  mov_actions = xthreat.get_successful_move_actions(gamestate_actions)
  xt_model = load_model('https://karun.in/blog/data/open_xt_12x8_v1.json')
  mov_actions['xt'] = xt_model.predict(mov_actions)
  return mov_actions[['game_id', 'action_id', 'xt']]

#%%
## compute from globals
selected_competitions = pd.DataFrame.from_dict({
  'competition_id': [COMPETITION_ID],
  'season_id': [SEASON_ID],
})

loader = OptaLoader(
  root=RAW_DATA_DIR,
  parser='whoscored',
  feeds={'whoscored': '{competition_id}\{season_id}\{game_id}.json'}
)

games = (
  get_games(loader)
  .sort_values(['competition_id', 'game_date', 'game_id'])
  .reset_index(drop=True)
)

#%%
get_bodyparts()
get_results()
get_actiontypes()
get_actiontypes_atomic()

teams = get_teams(loader=loader, games=games)
players = get_players(loader=loader, games=games) ## problem with MLS2020?
actions = get_actions(loader=loader, games=games) ## problem with MLS2020?
actions_atomic = get_actions_atomic(loader=loader, games=games)
gamestate_actions = get_gamestate_actions(loader=loader, games=games)
gamestate_action_game_ids = gamestate_actions.game_id.unique()
games_with_gamestate_actions = games.query('game_id in @gamestate_action_game_ids')
x = get_x(loader=loader, games=games_with_gamestate_actions)
x_atomic = get_x_atomic(loader=loader, games=games_with_gamestate_actions)
y = get_y(loader=loader, games=games_with_gamestate_actions)
y_atomic = get_y_atomic(loader=loader, games=games_with_gamestate_actions)
xt = get_xt(gamestate_actions) 

#%%
