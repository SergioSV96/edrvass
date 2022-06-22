import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer

from itertools import product

import matplotlib.pyplot as plt

import os, os.path



def get_nans_experiments_summary(experiments_summary):
    '''Get number of NaNs (0 values) for gravity, wind power and gravity and wind power columns.
    
    Parameters
    ----------
    experiments_summary : DataFrame
        DataFrame with the experiments summary.

    Returns
    ----------
    gravity_nans : int
        Number of NaNs in gravity column.
    wind_power_nans : int
        Number of NaNs in wind power column.
    gravity_wind_power_nans : int
        Number of NaNs in gravity and wind power columns.
    '''
    gravity_nans = (experiments_summary['gravity'] == 0).sum()
    wind_power_nans = (experiments_summary['wind_power'] == 0).sum()
    gravity_wind_power_nans = ((experiments_summary[['gravity', 'wind_power']] == 0).sum(axis=1) == 2).sum()

    return gravity_nans, wind_power_nans, gravity_wind_power_nans


def get_nans_per_columns_experiments_series(series_path):
    '''Get NaNs per column for each experiment in the series
    Three possible values are considered as NaNs: 0, NaN, -999
    
    Parameters
    ----------
    series_path : str
        Path to the series folder.

    Returns
    ----------
    nans_per_columns : DataFrame
        DataFrame with the number of NaNs per column for each experiment.
    '''
    nans_per_columns = []

    for filename in os.listdir(series_path):
        series_file = os.path.join(series_path, filename)
        experiment_series = pd.read_csv(series_file)

        nulls_serie = dict()

        for col in experiment_series.columns:
            nulls_serie[(col, 'nulls')] = experiment_series[col].isnull().sum().sum()
            nulls_serie[(col, '-999')] = (experiment_series[col] == -999).sum().sum()
            nulls_serie[(col, '0')] = (experiment_series[col] == 0).sum().sum()

        multi_columns = pd.MultiIndex.from_tuples( list(product(experiment_series.columns, ['nulls', '-999', '0'])) )
        data = pd.DataFrame(nulls_serie, index=[filename], columns=multi_columns)
        nans_per_columns.append(data)

    return pd.concat(nans_per_columns, axis=0)


def test_imputation_series(series_filepath, col, missing_value, imputation_method, interpolation_technique=None, order=None):
    '''Test imputation method for a given column with a missing value in a given series.
    Plots the original series and the imputed series.
    Valid imputation methods are: 'interpolation', 'IterativeImputer'

    If imputation method is 'interpolation', valid interpolation techniques are: 'linear', 'quadratic', 'cubic'
    If imputation method is 'interpolation' and interpolation technique is not 'linear', order must be specified.
    If imputation method is IterativeImputer, the interpolation technique and order are not required.

    Parameters
    ----------
    series_filepath : str
        Path to the series file.
    col : str
        Column name.
    missing_value : float or int or np.nan
        Missing value to be imputed.
    imputation_method : str
        Imputation method.
    interpolation_technique : str, optional
        Interpolation technique. The default is None.
    order : int, optional
        Order of the interpolation. The default is None.

    '''

    serie = pd.read_csv(series_filepath)

    fig = plt.figure(figsize=(9, 3))
    plt.plot(serie[col])
    plt.title(col + ' - ' + series_filepath.split('_')[1])
    plt.show()

    serie[col] = serie[col].replace(missing_value, np.nan)

    if imputation_method=='interpolation':
        if order:
            serie[col] = serie[col].interpolate(method=interpolation_technique, order=order)
        else:
            serie[col] = serie[col].interpolate(method=interpolation_technique)
    elif imputation_method=='IterativeImputer':
        imputer = IterativeImputer(missing_values=missing_value)
        serie[col] = imputer.fit_transform(serie[col].values.reshape(-1, 1))
    else:
        raise ValueError('Unknown imputation method')

    fig = plt.figure(figsize=(9, 3))
    plt.plot(serie[col])
    plt.title(col + ' - ' + series_filepath.split('_')[1] + ' - ' + imputation_method + ' ' + interpolation_technique if interpolation_technique else '')
    plt.show()

    print()


def get_error_imputed(experiments_summary, error_function, imputer=IterativeImputer()):
    '''Calculates the error of the imputed values for columns gravity and wind power for experiments summary.
    To calculate the error, original proportion of missing values in experiments is used for gravity, wind power and gravity and wind power columns.
    These can be individually missing or missing at the same time.

    Imputer can be any function that takes a series and returns a series.

    Error function must be: 'mae', 'rmse'.

    Parameters
    ----------
    experiments_summary : DataFrame
        DataFrame with the experiments summary.
    error_function : str
        Error function.
    imputer : function, optional
        Imputation function. The default is IterativeImputer().
    
    Returns
    ----------
    error_gravity : float
        Error of the imputed values for gravity column.
    error_wind_power : float
        Error of the imputed values for wind power column.
    error_gravity_wind_power : np.array
        Error of the imputed values for gravity and wind power columns.
    error_gravity_wind_power : np.array
        Error of the imputed values for gravity and wind power columns.
    '''
    experiments_summary_copy = experiments_summary.copy()
    experiments_summary_copy.drop(columns=['filename'], inplace=True)

    indexes_no_missing = experiments_summary_copy[(experiments_summary_copy['gravity'] != 0) & (experiments_summary_copy['wind_power'] != 0)].index

    # Original proportion of missing values for the columns is used to simulate the imputation
    gravity_nans, wind_power_nans, gravity_wind_power_nans = get_nans_experiments_summary(experiments_summary_copy)
    gravity_nans_percent, wind_power_nans_percent, gravity_wind_power_nans_percent = gravity_nans/experiments_summary_copy.shape[0], wind_power_nans/experiments_summary_copy.shape[0], gravity_wind_power_nans/experiments_summary_copy.shape[0]

    indexes_gravity_nans = np.random.choice(indexes_no_missing, int(len(indexes_no_missing) * gravity_nans_percent), replace=False)
    indexes_no_missing = np.setdiff1d(indexes_no_missing, indexes_gravity_nans)

    indexes_wind_power_nans = np.random.choice(indexes_no_missing, int(len(indexes_no_missing) * wind_power_nans_percent), replace=False)
    indexes_no_missing = np.setdiff1d(indexes_no_missing, indexes_wind_power_nans)

    indexes_gravity_wind_power_nans = np.random.choice(indexes_no_missing, int(len(indexes_no_missing) * gravity_wind_power_nans_percent), replace=False)
    indexes_no_missing = np.setdiff1d(indexes_no_missing, indexes_gravity_wind_power_nans)


    experiments_summary_copy.loc[indexes_gravity_nans, 'gravity'] = np.nan
    experiments_summary_copy.loc[indexes_wind_power_nans, 'wind_power'] = np.nan
    experiments_summary_copy.loc[indexes_gravity_wind_power_nans, ['gravity', 'wind_power']] = np.nan


    imputed_series = imputer.fit_transform(experiments_summary_copy.values)
    imputed_series = pd.DataFrame(imputed_series, columns=experiments_summary_copy.columns)
    imputed_series = imputed_series.set_index(experiments_summary_copy.index)

    if error_function == 'rmse':
        rmse_gravity = np.sqrt(np.mean((imputed_series.loc[indexes_gravity_nans, 'gravity'] - experiments_summary.loc[indexes_gravity_nans, 'gravity'])**2))
        rmse_wind_power = np.sqrt(np.mean((imputed_series.loc[indexes_wind_power_nans, 'wind_power'] - experiments_summary.loc[indexes_wind_power_nans, 'wind_power'])**2))
        rmse_gravity_wind_power = np.sqrt(np.mean((imputed_series.loc[indexes_gravity_wind_power_nans, ['gravity', 'wind_power']] - experiments_summary.loc[indexes_gravity_wind_power_nans, ['gravity', 'wind_power']])**2))
        rmse_total = np.sqrt(np.mean((imputed_series.loc[:, ['gravity', 'wind_power']] - experiments_summary.loc[:, ['gravity', 'wind_power']])**2))

        return rmse_gravity, rmse_wind_power, rmse_gravity_wind_power.values, rmse_total.values

    elif error_function == 'mae':
        mae_imputed_gravity_nans = np.mean(np.abs(imputed_series.loc[indexes_gravity_nans, 'gravity'] - experiments_summary.loc[indexes_gravity_nans, 'gravity']))
        mae_imputed_wind_power_nans = np.mean(np.abs(imputed_series.loc[indexes_wind_power_nans, 'wind_power'] - experiments_summary.loc[indexes_wind_power_nans, 'wind_power']))
        mae_imputed_gravity_wind_power_nans = np.mean(np.abs(imputed_series.loc[indexes_gravity_wind_power_nans, ['gravity', 'wind_power']] - experiments_summary.loc[indexes_gravity_wind_power_nans, ['gravity', 'wind_power']]))
        mae_imputed = np.mean(np.abs(imputed_series[['gravity', 'wind_power']] - experiments_summary[['gravity', 'wind_power']]))

        return mae_imputed_gravity_nans, mae_imputed_wind_power_nans, mae_imputed_gravity_wind_power_nans.values, mae_imputed.values
        
    else:
        raise ValueError('error_function must be either rmse or mae')


def test_imputers_error(experiments_summary):
    '''Calculate and prints the root mean squared error and mean absolute error for multiple imputation methods.
    Imputation methods are: KNNImputer(n_neighbors=15), SimpleImputer(strategy='mean'), SimpleImputer(strategy='median'), IterativeImputer().

    Parameters
    ----------
    experiments_summary : DataFrame
        DataFrame with the experiments summary.
    '''
    imputers = [KNNImputer(n_neighbors=15), SimpleImputer(strategy='median'), SimpleImputer(strategy='mean'), IterativeImputer()]

    for imputer in imputers:
        print(imputer)

        rmse_gravity, rmse_wind_power, rmse_gravity_wind_power, rmse_total = get_error_imputed(experiments_summary, error_function='rmse', imputer=imputer)
        print('RMSE:', rmse_gravity, rmse_wind_power, rmse_gravity_wind_power, rmse_total)

        mae_gravity_nans, mae_wind_power_nans, mae_gravity_wind_power_nans, mae_total = get_error_imputed(experiments_summary, error_function='mae', imputer=imputer)
        print('MAE:', mae_gravity_nans, mae_wind_power_nans, mae_gravity_wind_power_nans, mae_total)

        print()