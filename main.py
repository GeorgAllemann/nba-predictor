
import os
#from nba_api.stats.endpoints import teamgamelog
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix


def ensureDataInLocalFiles(start_year, end_year): 
  prepareDataDirectory()
  
  for season in generateSeasonParameters(start_year, end_year):
    writeSeasonToFile(season)
  
  return True

def writeSeasonToFile(season):
  full_filepath = os.path.join(os.path.dirname(__file__), 'data', 'season-{}.csv'.format(season)) 
  if not os.path.exists(full_filepath):
    league_gamelog = leaguegamelog.LeagueGameLog(season_all_time=season)
    games = league_gamelog.get_normalized_dict()['LeagueGameLog']
    csv_file = open(full_filepath, 'w', newline='')
    print(games)
    csv_writer = csv.DictWriter(csv_file, games[0].keys())
    csv_writer.writeheader()
    csv_writer.writerows(games)
    #w = csv.DictWriter(f,my_dict.keys())
    #w.writerows(my_dict)
  
  return True


def prepareDataDirectory():
  dirname = os.path.dirname(__file__)
  data_directory_name = os.path.join(dirname, 'data')
  if not os.path.exists(data_directory_name):
    os.mkdir(data_directory_name)
  
  return True

def generateSeasonParameters(start_year, end_year=None):
  if end_year == None:
    end_year = start_year
  if start_year > end_year:
    raise ValueError('Start year must be smaller than end year')

  seasons = []
  for year in range(start_year, end_year + 1):
    season = yearToSeason(year)
    seasons.append(season)

  return seasons

def yearToSeason(year):
  next_year = year + 1
  return '{}-{}'.format(year, str(next_year)[-2:])

def previousGamesMeans(previous_games):
  if len(previous_games.index) >= 10:
    results = previous_games[['FT_PCT', 'FG_PCT', 'REB', 'FG3_PCT', 'PLUS_MINUS']].mean()
    game_means = []
    for mean in results:
      game_means.append(mean)
    return game_means
  else:
    return []

def previousVictories(previous_games):
  return previous_games[previous_games['WL'] == 'W'].shape[0]

ensureDataInLocalFiles(2014, 2017)

X_train = []
Y_train = []
for season_parameters in generateSeasonParameters(2014, 2016):
  print(season_parameters)

  season = pd.read_csv('./data/season-{}.csv'.format(season_parameters))
  for index, game_pair in season.groupby('GAME_ID'):
    y = None
    x = []

    home_stats = []
    away_stats = []
    for index, game in game_pair.iterrows():
      previous_games = season[(season['TEAM_ABBREVIATION'] == game['TEAM_ABBREVIATION']) & (season['GAME_DATE'] < game['GAME_DATE'])].sort_values(by='GAME_DATE', ascending=False).head(10)
      if '@' not in game['MATCHUP']:
        y = int(game['WL'] == 'W')
        home_stats.extend(previousGamesMeans(previous_games))
        home_stats.append(previousVictories(previous_games))
      else :
        away_stats.extend(previousGamesMeans(previous_games))
        away_stats.append(previousVictories(previous_games))
      
    x = [*home_stats, *away_stats]
    
    if len(x) > 7:
      X_train.append(x)
      Y_train.append(y)


X_test = []
Y_test = []
test_season = pd.read_csv('./data/season-2017-18.csv')
for index, game_pair in test_season.groupby('GAME_ID'):
  y = None
  x = []

  home_stats = []
  away_stats = []
  for index, game in game_pair.iterrows():
    previous_games = test_season[(season['TEAM_ABBREVIATION'] == game['TEAM_ABBREVIATION']) & (test_season['GAME_DATE'] < game['GAME_DATE'])].sort_values(by='GAME_DATE', ascending=False).head(10)
    if '@' not in game['MATCHUP']:
      y = int(game['WL'] == 'W')
      home_stats.extend(previousGamesMeans(previous_games))
      home_stats.append(previousVictories(previous_games))
    else :
      away_stats.extend(previousGamesMeans(previous_games))
      away_stats.append(previousVictories(previous_games))

    x = [*home_stats, *away_stats]
  
  if len(x) > 7:
    X_test.append(x)
    Y_test.append(y)



scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


mlp = MLPClassifier(hidden_layer_sizes=(18,13,13),max_iter=5000)
mlp.fit(X_train,Y_train)

predictions = mlp.predict(X_test)
print(predictions)
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
