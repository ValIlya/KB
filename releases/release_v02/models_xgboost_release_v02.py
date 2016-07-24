# coding: utf-8
from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
import os, sys
import cPickle as pickle
import datetime
import xgboost as xgb

# hook up all scripts
sys.path.append(os.path.abspath('../../scripts')) # if notebook in folder
import preprocessing_time_series as pts

pd.options.mode.chained_assignment = None # turn off warnings

town = pts.text_encoding(pts.town_preproc())
data_test = pd.read_csv(pts.working_dir() + 'test.csv', index_col=0)
indexers = ['Semana', 'Agencia_ID', 'Canal_ID',
            'Ruta_SAK', 'Cliente_ID', 'Producto_ID']

def data_brush_for_model(state):
    data_train = pd.read_csv('Feature_releases/release_v02/train_%s.csv' % state)
    # only state from train set
    # agencies = set(town.loc[town.State == state].index)
    #
    # data_test_state = data_test.loc[data_test.Agencia_ID.isin(agencies), :]
    # data = pd.concat([data_train, data_test_state], axis=0)
    data = data_train # test set already included
    means_by_split = [x for x in data.columns if
                      'Mean' in x and not (x[-2] == '_' or x[-4:] == 'mean')]  # its mean, its not a lag
    data = data.drop(['Log_Dev_proxima', 'Log_Dev_uni_proxima',
                      'Log_Venta_hoy', 'Venta_uni_hoy', 'Venta_hoy',
                      'Dev_proxima', 'Demanda_uni_equil', 'Dev_proxima_by_uni', 'No_remains',
                      'Venta_hoy_by_uni'] + means_by_split, axis=1).set_index('Semana')

    return data

def model_building(data):

    X_train = data.loc[~data.Log_Demanda.isnull(),:].drop('Log_Demanda', axis=1)
    y_train = data.loc[~data.Log_Demanda.isnull(),'Log_Demanda']
    X_eval = data.loc[data.Log_Demanda.isnull(),:].drop('Log_Demanda', axis=1)
    # for 11 week special
    X_train2 = X_train[[col for col in X_train.columns if '_1' != col[-2:]]]
    X_eval2 = X_eval[[col for col in X_eval.columns if '_1' != col[-2:]]]

    param1 = {
        'learning_rate': 0.3,
        'gamma': 0,
        'max_depth': 10,
        'min_child_weight': 22,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    param2 = {
        'learning_rate': 0.3,
        'gamma': 0,
        'max_depth': 10,
        'min_child_weight': 24,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    xgb_model = xgb.XGBRegressor()
    xgb_model.set_params(**param1)
    xgb_model.fit(X_train, y_train)

    xgb_model2 = xgb.XGBRegressor()
    xgb_model2.set_params(**param2)
    xgb_model2.fit(X_train2, y_train)

    y_eval10 = X_eval.loc[10, indexers[1:]]
    y_eval10['Log_Demanda'] = xgb_model.predict(X_eval.loc[10, :])
    y_eval11 = X_eval.loc[11, indexers[1:]]
    y_eval11['Log_Demanda'] = xgb_model2.predict(X_eval2.loc[11, :])
    y_eval = pd.concat([y_eval10, y_eval11], axis=0)

    return y_eval, xgb_model, xgb_model2




if __name__ == '__main__':
    os.chdir(pts.working_dir())

    if not os.path.exists('Predictions'):
        os.makedirs('Predictions')
    if not os.path.exists('Predictions/release_v02'):
        os.makedirs('Predictions/release_v02')
    if not os.path.exists('Predictions/models_v02'):
        os.makedirs('Predictions/models_v02')


    start_time = datetime.datetime.now()

    states = town.State.unique()
    print(states)
    for i, state in enumerate(states):
        data = data_brush_for_model(state)
        print(state, 'read')
        y_eval, m1, m2 = model_building(data)

        y_eval.to_csv('Predictions/release_v02/Prediction_%s_v02.csv' % state)

        with open('Predictions/models_v02/xgboost_week10_%s.pkl' % state, 'wb') as model1, \
             open('Predictions/models_v02/xgboost_week11_%s.pkl' % state, 'wb') as model2:
            pickle.dump(m1, model1, 2)
            pickle.dump(m2, model2, 2)

        print('%s saved, %d to go, time:' % (state, len(states) - i - 1), datetime.datetime.now() - start_time)

    print('Final submit generating')
    state_files = ['Predictions/release_v02/Prediction_%s_v02.csv' % state for state in states]
    test_states = pd.concat([pd.read_csv(f) for f in state_files])
    test_data = pd.merge(data_test, test_states.set_index(indexers), 'inner',
                         left_on=indexers, right_index=True)
    assert test_data.shape[0] == data_test.shape[0], 'Not all id were predicted'
    test_data['Demanda_uni_equil'] = test_data.Log_Demanda.apply(np.expm1)
    test_data.index.name = 'id'
    test_data.loc[test_data['Demanda_uni_equil'] < 0, 'Demanda_uni_equil'] = 0
    test_data[['Demanda_uni_equil']].to_csv('Predictions/release_v02/Prediction_v02_nonnegative.csv', float_format='%.5f')
    print('Final submit saved')