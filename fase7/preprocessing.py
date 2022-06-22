import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer

import os, os.path



def get_serie_describe(serie):
    '''Returns a dictionary with its pandas describe() method values

    Parameters
    ----------
    serie : pandas.DataFrame
        The time serie to describe

    Returns
    ----------
    values : dict
        The dictionary with the describe() method values
    '''
    serie_describe = serie.describe().drop(columns=['leg_1', 'leg_2'])
    serie_describe = serie_describe.transpose().drop(columns=['count'])

    values = dict()

    for index, row in serie_describe.iterrows():
        for col in row.index:
            values[f'{index}_{col}'] = row[col]

    return values


def get_experiments_stats(series_path, experiments_summary_path):
    '''Calculate the statistics of the experiments and return a dataframe with them.
    The statistics columns are the given by describe() method of the pandas DataFrame and the
    following hand-crafted ones:
    - x0_pos: initial x position
    - y0_pos: initial y position
    - xf_pos: final x position
    - yf_pos: final y position
    - total_main_booster: cumulative sum of the main booster
    - total_lat_booster: cumulative sum of the lateral booster
    - legs: sum of number of legs (leg_1 + leg_2) in contact with the ground
    - total_dist: cumulative sum of the distance travelled
    - initial_end_dist: distance between the initial and the final position  

    Parameters
    ----------
    series_path : str
        The path to the folder containing the csv files with the series.
    
    Returns
    ----------
    experiments_summary : pandas.DataFrame
        The dataframe with the statistics of the experiments.
    '''
    experiments_stats = dict()
    experiments_summary = pd.read_excel(experiments_summary_path)

    for filename in experiments_summary['filename']:
        series_file = os.path.join(series_path, filename)
        experiment_series = pd.read_csv(series_file)

        experiments_stats[filename] = get_serie_describe(experiment_series)

        experiments_stats[filename] = {**experiments_stats[filename],
            'x0_pos': experiment_series.iloc[0]['x_pos'],
            'y0_pos': experiment_series.iloc[0]['y_pos'],
            'xf_pos': experiment_series.iloc[-1]['x_pos'],
            'yf_pos': experiment_series.iloc[-1]['y_pos'],
            'total_main_booster': experiment_series['main_booster'].abs().sum(),
            'total_lat_booster': experiment_series['lat_booster'].abs().sum(),
            'legs': experiment_series.iloc[-1]['leg_1'] + experiment_series.iloc[-1]['leg_2'],
            'total_dist': experiment_series[['x_pos', 'y_pos']].apply(lambda row: np.sqrt(row[0]**2 + row[1]**2), axis=1).sum(),
            'initial_end_dist': np.linalg.norm(experiment_series.iloc[0][['x_pos', 'y_pos']] - experiment_series.iloc[-1][['x_pos', 'y_pos']])
        }

    experiments_stats = pd.DataFrame.from_dict(experiments_stats, orient='index')
    experiments_stats = experiments_stats.reset_index()
    columns = ['filename'] + experiments_stats.columns[1:].tolist()
    experiments_stats.columns = columns
    return experiments_stats


def impute_summary_only(experiments_summary, imputer=IterativeImputer()):
    '''Impute the missing values of the experiments_summary dataframe.
    Missing values are 0s in gravity and wind_power columns.

    Columns used for the imputation are the original ones in summary and the given by describe() method of the pandas DataFrame.

    Parameters
    ----------
    experiments_summary : pandas.DataFrame
        The dataframe with experiments summary.
    imputer : sklearn.impute.base.Imputer
        The imputer to use.
    
    Returns
    ----------
    experiments_summary : pandas.DataFrame
        The dataframe with imputed values.
    '''
    experiments_summary_imputed = experiments_summary.loc[:, 'total_timesteps':].copy()

    experiments_summary_imputed.loc[experiments_summary_imputed['gravity'] == 0, 'gravity'] = np.nan
    experiments_summary_imputed.loc[experiments_summary_imputed['wind_power'] == 0, 'wind_power'] = np.nan

    imputed_series = imputer.fit_transform(experiments_summary_imputed.values)
    imputed_series = pd.DataFrame(imputed_series, columns=experiments_summary_imputed.columns)
    imputed_series = imputed_series.set_index(experiments_summary_imputed.index)

    experiments_summary_copy = experiments_summary.copy()
    experiments_summary_copy.loc[:, 'total_timesteps':] = imputed_series

    return experiments_summary_copy


def transform_data(series_path, experiments_summary_path, target_variable=None):
    '''Transform the data from the csv files to a numpy array.
    The data is transformed to a numpy array of shape (samples, timesteps, features).

    Time series missing values are imputed using linear interpolation.
    These missing values are:
    - -999 for the x_pos column
    - 0 for the y_pos column
    - np.nan for the y_vel column

    If the target variable is not None, the target variable is also transformed to a numpy array of shape (samples, timesteps, 1).
    
    Parameters
    ----------
    series_path : str
        The path to the folder containing the csv files with the series.
    experiments_summary_path : str
        The path to the experiments summary xlsx file.
    target_variable : str, optional
        The name of the target variable. The default is None.

    Returns
    ----------
    X : numpy.ndarray
        The numpy array with the data.
    y : numpy.ndarray, optional
        The numpy array with the target variable.
    '''
    experiments_summary = pd.read_excel(experiments_summary_path)
    experiments_summary = experiments_summary.set_index('filename')

    X = []
    y = []
    for filename in experiment_summary.index:
        series_file = os.path.join(series_path, filename)
        experiment_series = pd.read_csv(series_file)
        
        experiment_summary = experiments_summary.loc[filename]
        
        # Preprocessing steps
        # - Add static fields
        # for static_var in ['gravity', 'wind_power', 'turbulence_power']:
        #    experiment_series[static_var] = np.repeat(experiment_summary[static_var], experiment_summary['total_timesteps']+1)
        
        # Imputation of NaNs
        for col, missing_value in [('x_pos', -999), ('y_pos', 0), ('y_vel', np.nan)]:
            experiment_series[col] = experiment_series[col].replace(missing_value, np.nan)
            experiment_series[col] = experiment_series[col].interpolate(method='linear', limit_direction='both')
        
        X.append(experiment_series.values)

        if target_variable is not None:
            y.append(experiment_summary[target_variable])
    
    if target_variable is not None:
        return np.array(X), np.array(y)
    else:
        return np.array(X)


def max_timesteps_length(X):
    '''Get the maximum length of the given data.
    '''
    return max([x.shape[0] for x in X])


def pad_data(X, max_length):
    '''Pad the data to make it of the given length.
    The data is padded with zeros.

    Parameters
    ----------
    X : numpy.ndarray
        The numpy array with the data.
    max_length : int
        The maximum length of the data.
    
    Returns
    ----------
    numpy.ndarray
        The numpy array with the padded data.
    '''
    X_padded = []
    for x in X:
        x_padded = np.zeros((max_length, x.shape[1]))
        x_padded[:x.shape[0], :] = x
        X_padded.append(x_padded)
    return np.array(X_padded)


def reshape_data_lstm(X):
    '''Reshape the data to be compatible with the LSTM.

    Parameters
    ----------
    X : numpy.ndarray
        The numpy array with the data.
    
    Returns
    ----------
    numpy.ndarray
        The numpy array with the reshaped data.
    '''
    return X.reshape((X.shape[0], X[0].shape[0], X[0].shape[1]))