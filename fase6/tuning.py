from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, make_scorer, recall_score, precision_score
from sklearn.decomposition import PCA
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin

from xgboost import XGBClassifier

import numpy as np
import pandas as pd
import math

import copy

import mlflow

from utils import retrieve_name



### PREPROCESSORS

preprocessor = ColumnTransformer([
        # (name, transformer, columns)
        ("temperatureFirstHalfPlanetRotation", make_pipeline(SimpleImputer(missing_values=-999.0, strategy='median'), FunctionTransformer(lambda f: (f - 32) / 1.8, feature_names_out="one-to-one"), StandardScaler()), ['temperatureFirstHalfPlanetRotation']), # convert from Fahrenheit to Celsius
        ("temperatureSecondHalfPlanetRotation", StandardScaler(), ['temperatureSecondHalfPlanetRotation']), # pass through the column unchanged
        ("waterStreamDistanceX", make_pipeline(FunctionTransformer(lambda f: f * 0.3048, feature_names_out="one-to-one"), StandardScaler()), ['waterStreamDistanceX']), # convert from feet to meters
        ("waterStreamDistanceY", StandardScaler(), ['waterStreamDistanceY']), # pass through the column unchanged
        ("planetSection", OneHotEncoder(handle_unknown = "ignore"), ['planetSection']), # one-hot encode the planetSection column
        ("cover", OneHotEncoder(handle_unknown='error', drop='first'), ['cover']), # one-hot encode the cover column and drop the first column (the one with the missing values == 0)
        ("climaticZone", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ['climaticZone']), # ordinal encode the climaticZone column TODO: drop category 3? what to do? only one row has a 3
        ("geoZone", OneHotEncoder(handle_unknown = "ignore"), ['geoZone']), # one-hot encode the geoZone column TODO: drop category 5?
        ("rockSize", OneHotEncoder(handle_unknown='ignore', drop='first'), ['rockSize']), # one-hot encode the rockSize column and drop the first column (the one with the missing values == 0)
        ("magmaConcentrationDistance", OneHotEncoder(handle_unknown = "ignore"), ['magmaConcentrationDistance']), # one-hot encode the rockSize column and drop the first column (the one with the missing values == 0) TODO: use Ordinal Encoder?
        ("mineralDensity", make_pipeline(SimpleImputer(missing_values=-999.0, strategy='median'), StandardScaler()), ['mineralDensity']), # pass through the column unchanged
        ("detectionDepth", StandardScaler(), ['detectionDepth']), # pass through the column unchanged TODO: convert km to m?
        ("longitude", StandardScaler(), ['longitude']), # pass through the column unchanged TODO: values > 360? do x - 360
    ],
    verbose_feature_names_out=False, remainder='passthrough'
)

preprocessor_all_robust = ColumnTransformer([
        # (name, transformer, columns)
        ("temperatureFirstHalfPlanetRotation", make_pipeline(FunctionTransformer(lambda f: (f - 32) / 1.8, feature_names_out="one-to-one"), RobustScaler()), ['temperatureFirstHalfPlanetRotation']), # convert from Fahrenheit to Celsius
        ("temperatureSecondHalfPlanetRotation", RobustScaler(), ['temperatureSecondHalfPlanetRotation']), # pass through the column unchanged
        ("waterStreamDistanceX", make_pipeline(FunctionTransformer(lambda f: f * 0.3048, feature_names_out="one-to-one"), RobustScaler()), ['waterStreamDistanceX']), # convert from feet to meters
        ("waterStreamDistanceY", RobustScaler(), ['waterStreamDistanceY']), # pass through the column unchanged
        ("planetSection", OneHotEncoder(handle_unknown = "ignore"), ['planetSection']), # one-hot encode the planetSection column
        ("cover", OneHotEncoder(handle_unknown='error', drop='first'), ['cover']), # one-hot encode the cover column and drop the first column (the one with the missing values == 0)
        ("climaticZone", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ['climaticZone']), # ordinal encode the climaticZone column TODO: drop category 3? what to do? only one row has a 3
        ("geoZone", OneHotEncoder(handle_unknown = "ignore"), ['geoZone']), # one-hot encode the geoZone column TODO: drop category 5?
        ("rockSize", OneHotEncoder(handle_unknown='ignore', drop='first'), ['rockSize']), # one-hot encode the rockSize column and drop the first column (the one with the missing values == 0)
        ("magmaConcentrationDistance", OneHotEncoder(handle_unknown = "ignore"), ['magmaConcentrationDistance']), # one-hot encode the rockSize column and drop the first column (the one with the missing values == 0) TODO: use Ordinal Encoder?
        ("mineralDensity", RobustScaler(), ['mineralDensity']), # pass through the column unchanged
        ("detectionDepth", RobustScaler(), ['detectionDepth']), # pass through the column unchanged TODO: convert km to m?
        ("longitude", RobustScaler(), ['longitude']), # pass through the column unchanged TODO: values > 360? do x - 360
    ],
    verbose_feature_names_out=False, remainder='passthrough'
)

preprocessor_mixed = ColumnTransformer([
        # (name, transformer, columns)
        ("temperatureFirstHalfPlanetRotation", make_pipeline(FunctionTransformer(lambda f: (f - 32) / 1.8, feature_names_out="one-to-one"), RobustScaler()), ['temperatureFirstHalfPlanetRotation']), # convert from Fahrenheit to Celsius
        ("temperatureSecondHalfPlanetRotation", StandardScaler(), ['temperatureSecondHalfPlanetRotation']), # pass through the column unchanged
        ("waterStreamDistanceX", make_pipeline(FunctionTransformer(lambda f: f * 0.3048, feature_names_out="one-to-one"), RobustScaler()), ['waterStreamDistanceX']), # convert from feet to meters
        ("waterStreamDistanceY", RobustScaler(), ['waterStreamDistanceY']), # pass through the column unchanged
        ("planetSection", OneHotEncoder(handle_unknown = "ignore"), ['planetSection']), # one-hot encode the planetSection column
        ("cover", OneHotEncoder(handle_unknown='error', drop='first'), ['cover']), # one-hot encode the cover column and drop the first column (the one with the missing values == 0)
        ("climaticZone", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ['climaticZone']), # ordinal encode the climaticZone column TODO: drop category 3? what to do? only one row has a 3
        ("geoZone", OneHotEncoder(handle_unknown = "ignore"), ['geoZone']), # one-hot encode the geoZone column TODO: drop category 5?
        ("rockSize", OneHotEncoder(handle_unknown='ignore', drop='first'), ['rockSize']), # one-hot encode the rockSize column and drop the first column (the one with the missing values == 0)
        ("magmaConcentrationDistance", OneHotEncoder(handle_unknown = "ignore"), ['magmaConcentrationDistance']), # one-hot encode the rockSize column and drop the first column (the one with the missing values == 0) TODO: use Ordinal Encoder?
        ("mineralDensity", StandardScaler(), ['mineralDensity']), # pass through the column unchanged
        ("detectionDepth", StandardScaler(), ['detectionDepth']), # pass through the column unchanged TODO: convert km to m?
        ("longitude", StandardScaler(), ['longitude']), # pass through the column unchanged TODO: values > 360? do x - 360
    ],
    verbose_feature_names_out=False, remainder='passthrough'
)

rockSizeMapping = {0:0, 1:2, 2:3, 3:1}
# preprocessor_fixed = ColumnTransformer([
#         # (name, transformer, columns)
#         ("temperatureFirstHalfPlanetRotation", make_pipeline(SimpleImputer(missing_values=-999.0, strategy='median'), FunctionTransformer(lambda f: (f - 32) / 1.8, feature_names_out="one-to-one"), RobustScaler()), ['temperatureFirstHalfPlanetRotation']),
#         ("temperatureSecondHalfPlanetRotation", StandardScaler(), ['temperatureSecondHalfPlanetRotation']),
#         ("waterStreamDistanceX", make_pipeline(FunctionTransformer(lambda f: f * 0.3048, feature_names_out="one-to-one"), RobustScaler()), ['waterStreamDistanceX']),
#         ("waterStreamDistanceY", RobustScaler(), ['waterStreamDistanceY']),
#         ("planetSection", OneHotEncoder(handle_unknown = "ignore"), ['planetSection']),
#         ("cover", OrdinalEncoder(handle_unknown='error'), ['cover']),
#         ("climaticZone", OneHotEncoder(handle_unknown="error"), ['climaticZone']),
#         ("geoZone", OneHotEncoder(handle_unknown = "error"), ['geoZone']),
#         ("rockSize", 'passthrough', ['rockSize']),
#         ("magmaConcentrationDistance", OneHotEncoder(handle_unknown = "ignore"), ['magmaConcentrationDistance']), # one-hot encode the rockSize column and drop the first column (the one with the missing values == 0) TODO: use Ordinal Encoder?
#         ("mineralDensity", StandardScaler(), ['mineralDensity']), # imputar valor de -999.0
#         ("detectionDepth", make_pipeline(FunctionTransformer(lambda f: f * 1000, feature_names_out="one-to-one"), StandardScaler()), ['detectionDepth']), # pass through the column unchanged TODO: convert km to m?
#         ("longitude", make_pipeline(FunctionTransformer(lambda f: f % 360, feature_names_out="one-to-one"), StandardScaler()), ['longitude']), # pass through the column unchanged TODO: values > 360? do x - 360
#     ],
#     verbose_feature_names_out=False, remainder='passthrough'
# )

preprocessor_fixed = ColumnTransformer([
        # (name, transformer, columns)
        ("temperatureFirstHalfPlanetRotation", make_pipeline(SimpleImputer(missing_values=-999.0, strategy='median'), FunctionTransformer(lambda f: (f - 32) / 1.8, feature_names_out="one-to-one"), StandardScaler()), ['temperatureFirstHalfPlanetRotation']),
        ("temperatureSecondHalfPlanetRotation", StandardScaler(), ['temperatureSecondHalfPlanetRotation']),
        ("waterStreamDistanceX", make_pipeline(FunctionTransformer(lambda f: f * 0.3048, feature_names_out="one-to-one"), StandardScaler()), ['waterStreamDistanceX']),
        ("waterStreamDistanceY", StandardScaler(), ['waterStreamDistanceY']),
        ("planetSection", OneHotEncoder(handle_unknown = "ignore"), ['planetSection']),
        ("cover", OrdinalEncoder(handle_unknown='error'), ['cover']),
        ("climaticZone", OrdinalEncoder(handle_unknown="error"), ['climaticZone']),
        ("geoZone", OneHotEncoder(handle_unknown = "error"), ['geoZone']),
        # ("rockSize", 'passthrough', ['rockSize']),
        ("rockSize", OneHotEncoder(handle_unknown = "error"), ['rockSize']),
        ("magmaConcentrationDistance", OneHotEncoder(handle_unknown = "ignore"), ['magmaConcentrationDistance']), # one-hot encode the rockSize column and drop the first column (the one with the missing values == 0) TODO: use Ordinal Encoder?
        # ("mineralDensity", StandardScaler(), ['mineralDensity']), # imputar valor de -999.0
        # ("mineralDensity", make_pipeline(SimpleImputer(missing_values=-999.0, strategy='median'), StandardScaler()), ['mineralDensity']),
        ("mineralDensity", make_pipeline(KNNImputer(missing_values=-999.0, n_neighbors=5, weights="distance"), StandardScaler()), ['mineralDensity']),
        # ("detectionDepth", make_pipeline(FunctionTransformer(lambda f: f * 1000, feature_names_out="one-to-one"), StandardScaler()), ['detectionDepth']),
        ("detectionDepth", StandardScaler(), ['detectionDepth']), # pass through the column unchanged TODO: convert km to m?
        ("longitude", make_pipeline(FunctionTransformer(lambda f: f % 360, feature_names_out="one-to-one"), StandardScaler()), ['longitude']),
        # ("longitude", StandardScaler(), ['longitude']), # pass through the column unchanged TODO: values > 360? do x - 360
    ],
    verbose_feature_names_out=False, remainder='passthrough'
)


preprocessor_fixed2 = ColumnTransformer([
        # (name, transformer, columns)
        ("temperatureFirstHalfPlanetRotation", make_pipeline(SimpleImputer(missing_values=-999.0, strategy='median'), FunctionTransformer(lambda f: (f - 32) / 1.8, feature_names_out="one-to-one"), StandardScaler()), ['temperatureFirstHalfPlanetRotation']),
        ("temperatureSecondHalfPlanetRotation", StandardScaler(), ['temperatureSecondHalfPlanetRotation']),
        ("waterStreamDistanceX", make_pipeline(FunctionTransformer(lambda f: f * 0.3048, feature_names_out="one-to-one"), StandardScaler()), ['waterStreamDistanceX']),
        ("waterStreamDistanceY", StandardScaler(), ['waterStreamDistanceY']),
        ("planetSection", OneHotEncoder(handle_unknown = "ignore"), ['planetSection']),
        # ("cover", OrdinalEncoder(handle_unknown='error'), ['cover']),
        ("cover", OneHotEncoder(handle_unknown='error', drop='first'), ['cover']),
        ("climaticZone", OrdinalEncoder(handle_unknown="error"), ['climaticZone']),
        ("geoZone", OneHotEncoder(handle_unknown = "error"), ['geoZone']),
        # ("rockSize", 'passthrough', ['rockSize']),
        ("rockSize", OneHotEncoder(handle_unknown = "error", drop='first'), ['rockSize']),
        ("magmaConcentrationDistance", OneHotEncoder(handle_unknown = "ignore"), ['magmaConcentrationDistance']), # one-hot encode the rockSize column and drop the first column (the one with the missing values == 0) TODO: use Ordinal Encoder?
        # ("mineralDensity", StandardScaler(), ['mineralDensity']), # imputar valor de -999.0
        # ("mineralDensity", make_pipeline(SimpleImputer(missing_values=-999.0, strategy='median'), StandardScaler()), ['mineralDensity']),
        ("mineralDensity", make_pipeline(SimpleImputer(missing_values=-999.0, strategy='constant', fill_value=0), StandardScaler()), ['mineralDensity']),
        # ("detectionDepth", make_pipeline(FunctionTransformer(lambda f: f * 1000, feature_names_out="one-to-one"), StandardScaler()), ['detectionDepth']),
        ("detectionDepth", StandardScaler(), ['detectionDepth']), # pass through the column unchanged TODO: convert km to m?
        ("longitude", StandardScaler(), ['longitude']),
        # ("longitude", StandardScaler(), ['longitude']), # pass through the column unchanged TODO: values > 360? do x - 360
    ],
    verbose_feature_names_out=False, remainder='passthrough'
)

best_preprocessor = ColumnTransformer([
        # (name, transformer, columns)
        ("temperatureFirstHalfPlanetRotation", make_pipeline(SimpleImputer(missing_values=-999.0, strategy='median'), FunctionTransformer(lambda f: (f - 32) / 1.8, feature_names_out="one-to-one"), StandardScaler()), ['temperatureFirstHalfPlanetRotation']),
        ("temperatureSecondHalfPlanetRotation", StandardScaler(), ['temperatureSecondHalfPlanetRotation']),
        ("waterStreamDistanceX", make_pipeline(FunctionTransformer(lambda f: f * 0.3048, feature_names_out="one-to-one"), StandardScaler()), ['waterStreamDistanceX']),
        ("waterStreamDistanceY", StandardScaler(), ['waterStreamDistanceY']),
        ("planetSection", OneHotEncoder(handle_unknown = "ignore"), ['planetSection']),
        ("cover", OneHotEncoder(handle_unknown='error', drop='first'), ['cover']),
        ("climaticZone", OrdinalEncoder(handle_unknown="error"), ['climaticZone']),
        ("geoZone", OneHotEncoder(handle_unknown = "error"), ['geoZone']),
        ("rockSize", OneHotEncoder(handle_unknown = "error", drop='first'), ['rockSize']),
        ("magmaConcentrationDistance", OneHotEncoder(handle_unknown = "ignore"), ['magmaConcentrationDistance']), # one-hot encode the rockSize column and drop the first column (the one with the missing values == 0) TODO: use Ordinal Encoder?
        # ("mineralDensity", StandardScaler(), ['mineralDensity']), # imputar valor de -999.0
        # ("mineralDensity", make_pipeline(SimpleImputer(missing_values=-999.0, strategy='median'), StandardScaler()), ['mineralDensity']),
        ("mineralDensity", make_pipeline(SimpleImputer(missing_values=-999.0, strategy='median'), StandardScaler()), ['mineralDensity']),
        # ("detectionDepth", make_pipeline(FunctionTransformer(lambda f: f * 1000, feature_names_out="one-to-one"), StandardScaler()), ['detectionDepth']),
        ("detectionDepth", StandardScaler(), ['detectionDepth']), # pass through the column unchanged TODO: convert km to m?
        ("longitude", StandardScaler(), ['longitude']),
        # ("longitude", StandardScaler(), ['longitude']), # pass through the column unchanged TODO: values > 360? do x - 360
    ],
    verbose_feature_names_out=False, remainder='passthrough'
)

magma_ordinal_preprocessor = ColumnTransformer([
        # (name, transformer, columns)
        ("temperatureFirstHalfPlanetRotation", make_pipeline(SimpleImputer(missing_values=-999.0, strategy='median'), FunctionTransformer(lambda f: (f - 32) / 1.8, feature_names_out="one-to-one"), StandardScaler()), ['temperatureFirstHalfPlanetRotation']),
        ("temperatureSecondHalfPlanetRotation", StandardScaler(), ['temperatureSecondHalfPlanetRotation']),
        ("waterStreamDistanceX", make_pipeline(FunctionTransformer(lambda f: f * 0.3048, feature_names_out="one-to-one"), StandardScaler()), ['waterStreamDistanceX']),
        ("waterStreamDistanceY", StandardScaler(), ['waterStreamDistanceY']),
        ("planetSection", OneHotEncoder(handle_unknown = "ignore"), ['planetSection']),
        ("cover", OneHotEncoder(handle_unknown='error', drop='first'), ['cover']),
        ("climaticZone", OrdinalEncoder(handle_unknown="error"), ['climaticZone']),
        ("geoZone", OneHotEncoder(handle_unknown = "error"), ['geoZone']),
        ("rockSize", OneHotEncoder(handle_unknown = "error", drop='first'), ['rockSize']),
        # ("magmaConcentrationDistance", OneHotEncoder(handle_unknown = "ignore"), ['magmaConcentrationDistance']),
        ("magmaConcentrationDistance", OrdinalEncoder(categories=[['VERY_CLOSE', 'CLOSE', 'MEDIUM', 'FAR', 'VERY_FAR']], handle_unknown = "error"), ['magmaConcentrationDistance']),
        # ("mineralDensity", StandardScaler(), ['mineralDensity']), # imputar valor de -999.0
        # ("mineralDensity", make_pipeline(SimpleImputer(missing_values=-999.0, strategy='median'), StandardScaler()), ['mineralDensity']),
        ("mineralDensity", make_pipeline(SimpleImputer(missing_values=-999.0, strategy='median'), StandardScaler()), ['mineralDensity']),
        # ("detectionDepth", make_pipeline(FunctionTransformer(lambda f: f * 1000, feature_names_out="one-to-one"), StandardScaler()), ['detectionDepth']),
        ("detectionDepth", StandardScaler(), ['detectionDepth']), # pass through the column unchanged TODO: convert km to m?
        ("longitude", StandardScaler(), ['longitude']),
        # ("longitude", StandardScaler(), ['longitude']), # pass through the column unchanged TODO: values > 360? do x - 360
    ],
    verbose_feature_names_out=False, remainder='passthrough'
)

preprocessor_robustScaler = ColumnTransformer([
        # (name, transformer, columns)
        ("temperatureFirstHalfPlanetRotation", make_pipeline(SimpleImputer(missing_values=-999.0, strategy='median'), FunctionTransformer(lambda f: (f - 32) / 1.8, feature_names_out="one-to-one"), RobustScaler()), ['temperatureFirstHalfPlanetRotation']),
        ("temperatureSecondHalfPlanetRotation", RobustScaler(), ['temperatureSecondHalfPlanetRotation']),
        ("waterStreamDistanceX", make_pipeline(FunctionTransformer(lambda f: f * 0.3048, feature_names_out="one-to-one"), RobustScaler()), ['waterStreamDistanceX']),
        ("waterStreamDistanceY", RobustScaler(), ['waterStreamDistanceY']),
        ("planetSection", OneHotEncoder(handle_unknown = "ignore"), ['planetSection']),
        ("cover", OneHotEncoder(handle_unknown='error', drop='first'), ['cover']),
        ("climaticZone", OrdinalEncoder(handle_unknown="error"), ['climaticZone']),
        ("geoZone", OneHotEncoder(handle_unknown = "error"), ['geoZone']),
        ("rockSize", OneHotEncoder(handle_unknown = "error", drop='first'), ['rockSize']),
        ("magmaConcentrationDistance", OneHotEncoder(handle_unknown = "ignore"), ['magmaConcentrationDistance']), # one-hot encode the rockSize column and drop the first column (the one with the missing values == 0) TODO: use Ordinal Encoder?
        # ("mineralDensity", StandardScaler(), ['mineralDensity']), # imputar valor de -999.0
        # ("mineralDensity", make_pipeline(SimpleImputer(missing_values=-999.0, strategy='median'), StandardScaler()), ['mineralDensity']),
        ("mineralDensity", make_pipeline(SimpleImputer(missing_values=-999.0, strategy='median'), RobustScaler()), ['mineralDensity']),
        # ("detectionDepth", make_pipeline(FunctionTransformer(lambda f: f * 1000, feature_names_out="one-to-one"), StandardScaler()), ['detectionDepth']),
        ("detectionDepth", RobustScaler(), ['detectionDepth']), # pass through the column unchanged TODO: convert km to m?
        ("longitude", RobustScaler(), ['longitude']),
        # ("longitude", StandardScaler(), ['longitude']), # pass through the column unchanged TODO: values > 360? do x - 360
    ],
    verbose_feature_names_out=False, remainder='passthrough'
)

ordinals_best_preprocessor = ColumnTransformer([
        # (name, transformer, columns)
        ("temperatureFirstHalfPlanetRotation", make_pipeline(SimpleImputer(missing_values=-999.0, strategy='median'), FunctionTransformer(lambda f: (f - 32) / 1.8, feature_names_out="one-to-one"), StandardScaler()), ['temperatureFirstHalfPlanetRotation']),
        ("temperatureSecondHalfPlanetRotation", StandardScaler(), ['temperatureSecondHalfPlanetRotation']),
        ("waterStreamDistanceX", make_pipeline(FunctionTransformer(lambda f: f * 0.3048, feature_names_out="one-to-one"), StandardScaler()), ['waterStreamDistanceX']),
        ("waterStreamDistanceY", StandardScaler(), ['waterStreamDistanceY']),
        ("planetSection", OneHotEncoder(handle_unknown = "ignore"), ['planetSection']),
        # ("cover", OneHotEncoder(handle_unknown='error', drop='first'), ['cover']),
        ("cover", 'passthrough', ['cover']), # DIFERENTE best_preprocessor
        ("climaticZone", 'passthrough', ['climaticZone']), # puede que el ordinal encoder sea indeferente, solo para bajar el rango
        ("geoZone", OneHotEncoder(handle_unknown = "error"), ['geoZone']),
        ("rockSize", OneHotEncoder(handle_unknown = "error", drop='first'), ['rockSize']), # da peores resultados utilizando ordinal
        ("magmaConcentrationDistance", OrdinalEncoder(categories=[['VERY_CLOSE', 'CLOSE', 'MEDIUM', 'FAR', 'VERY_FAR']], handle_unknown = "error"), ['magmaConcentrationDistance']), # invirtiendo el orden del encoding no hay diferencia
        ("mineralDensity", make_pipeline(SimpleImputer(missing_values=-999.0, strategy='median'), StandardScaler()), ['mineralDensity']),
        ("detectionDepth", make_pipeline(FunctionTransformer(lambda f: f * 1000, feature_names_out="one-to-one"), StandardScaler()), ['detectionDepth']), # no hay diferencia cambiando que sin cambiar de km a m
        ("longitude", make_pipeline(FunctionTransformer(lambda f: f % 360, feature_names_out="one-to-one"), StandardScaler()), ['longitude']) # no hay diferencia haciendo % 360
    ],
    verbose_feature_names_out=False, remainder='passthrough'
)

def euclidean_distance(x, y):
    res = []
    for i in range(len(x)):
        res.append(math.sqrt(x[i]**2 + y[i]**2))
    return np.array(res)


class CreateVariables(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y = None):
        # waterStreamDistanceX = X[:,2]
        # waterStreamDistanceY = X[:,3]
        # waterStreamDistance = euclidean_distance(waterStreamDistanceX, waterStreamDistanceY)
        # X = np.append(X, waterStreamDistance.reshape(-1, 1), axis=1)
        return self

    def transform(self, X, y = None):
        waterStreamDistanceX = X[:,2]
        waterStreamDistanceY = X[:,3]
        waterStreamDistance = euclidean_distance(waterStreamDistanceX, waterStreamDistanceY)
        X = np.append(X, waterStreamDistance.reshape(-1, 1), axis=1)
        return X


### PARAMETERS FOR GRID SEARCH

knn_params={
    'model__n_neighbors': [5, 8, 10, 12, 15, 20],
    'model__weights': ['uniform', 'distance'],
    'model__metric': ['euclidean', 'manhattan']
}

randomForest_params = { 
    'model__n_estimators': [200, 500],
    'model__max_features': ['auto', 'sqrt', 'log2'],
    'model__max_depth' : [5,6,7],
    'model__criterion' :['gini', 'entropy']
}

log_reg_params = {
    "model__C":np.logspace(-3,3,7),
    "model__penalty":["l2"]}

xgboost_params = {
    'model__min_child_weight': [1, 5, 10],
    'model__gamma': [0.5, 1, 1.5, 2, 5],
    'model__subsample': [0.6, 0.8, 1.0],
    'model__colsample_bytree': [0.6, 0.8, 1.0],
    'model__max_depth': [3, 4, 5]
}


### SCORES

scores = ['accuracy', 'precision_macro']
refit_metric = 'accuracy'


### MODELS PARAMETERS

# preprocessors = [preprocessor, preprocessor_all_robust, preprocessor_mixed]
preprocessors = [best_preprocessor, ordinals_best_preprocessor]
# models = [KNeighborsClassifier(), RandomForestClassifier()]
models = [KNeighborsClassifier()]
# params = {'KNeighborsClassifier':knn_params, 'RandomForestClassifier':randomForest_params}
# params = {'LogisticRegression':log_reg_params}
params = {'KNeighborsClassifier':knn_params}


### DATA
pd.set_option('display.max_columns', None)
train = pd.read_csv('fase6/data/train_data.csv')
test = pd.read_csv('fase6/data/test_data.csv')



X = train.copy()
y = train['mineralType'].map(lambda x: x-1)

y = y.drop(y[X['climaticZone']==3].index)
X = X.drop(X[X['climaticZone']==3].index)
y = y.drop(y[X['geoZone']==5].index.values)
X = X.drop(X[X['geoZone']==5].index.values)
X['rockSize'] = X['rockSize'].map(rockSizeMapping)

X = X.drop(['mineralType', 'id'], axis=1)



def main(X, y):

    # mlflow.sklearn.autolog()

    # for preprocessor in preprocessors:
    #     print(retrieve_name(preprocessor))

    #     for model in models:
    #         model_name = type(model).__name__
    #         print(model_name)

    #         mlflow.set_experiment(model_name)

    #         grid = GridSearchCV(model, cv=10, scoring=scores, error_score='raise', return_train_score=True, n_jobs=-1, verbose=4, refit=refit_metric,
    #             param_grid=params[model_name]
    #         )

    #         with mlflow.start_run(run_name=retrieve_name(preprocessor)) as run:
    #             grid.fit(preprocessor.fit_transform(X), y)

    
    for preprocessor in preprocessors:
        print(retrieve_name(preprocessor))

        for model in models:
            model_name = type(model).__name__
            print(model_name)

            mlflow.sklearn.autolog()

            mlflow.set_experiment(model_name)

            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('createVariables', CreateVariables()),
                ('pca', PCA()),
                ('model', model)
            ])

            grid = GridSearchCV(pipe, cv=10, scoring=scores, error_score='raise', return_train_score=True, n_jobs=-1, verbose=4, refit=refit_metric,
                param_grid=params[model_name]
            )

            # grid = RandomizedSearchCV(pipe, cv=15, scoring=scores, error_score='raise', return_train_score=True, n_jobs=-1, verbose=4, refit=refit_metric,
            #     param_distributions=params[model_name]
            # )

            with mlflow.start_run(run_name=retrieve_name(preprocessor)) as run:
                grid.fit(X, y)


main(X, y)