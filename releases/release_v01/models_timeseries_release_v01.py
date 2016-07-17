# coding: utf-8
from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
import os, sys
import cPickle as pickle
import datetime
from sklearn.ensemble import RandomForestRegressor
# hook up all scripts
sys.path.append(os.path.abspath('../../scripts')) # if notebook in folder
import preprocessing_time_series as pts

pd.options.mode.chained_assignment = None # turn off warnings

town = pts.text_encoding(pts.town_preproc())
data_test = pd.read_csv(pts.working_dir() + 'test.csv', index_col=0)

def data_brush_for_model(state):
    data_train = pd.read_csv('Feature_releases/release_v01/train_%s.csv' % state)
    # only state from train set
    agencies = set(town.loc[town.State == state].index)

    data_test_state = data_test.loc[data_test.Agencia_ID.isin(agencies), :]
    data = pd.concat([data_train, data_test_state], axis=0)
    data = pts.lag_generation(data, n_lags=5)
    data = data.drop([u'Town', u'State', u'Venta_uni_hoy', u'Venta_hoy', u'Dev_uni_proxima',
           u'Dev_proxima', u'Demanda_uni_equil', u'Dev_proxima_by_uni', u'No_remains',
            u'Venta_hoy_by_uni', u'Ordered',u'Median_Producto_ID',
           u'LogMean_Producto_ID', u'Median_Producto_ID_Ruta_SAK',
           u'LogMean_Producto_ID_Ruta_SAK',
           u'Median_Producto_ID_Cliente_ID_Agencia_ID',
           u'LogMean_Producto_ID_Cliente_ID_Agencia_ID'], axis=1).set_index(u'Semana', append=True)
    data = data.swaplevel(i=0, j=-1, axis=0)

    return data

def model_building(data):

    X_train = data.loc[~data.Log_Demanda.isnull(),:].drop('Log_Demanda', axis=1)
    y_train = data.loc[~data.Log_Demanda.isnull(),'Log_Demanda']
    X_eval = data.loc[data.Log_Demanda.isnull(),:].drop('Log_Demanda', axis=1)
    # for 11 week special
    X_train2 = X_train[[col for col in X_train.columns if '_1' != col[-2:]]]
    X_eval2 = X_eval[[col for col in X_eval.columns if '_1' != col[-2:]]]

    rf = RandomForestRegressor(n_estimators=3, criterion='mse', n_jobs=-1, verbose=10)
    rf.fit(X_train.fillna(-999), y_train)
    rf2 = RandomForestRegressor(n_estimators=3, criterion='mse', n_jobs=-1, verbose=10)
    rf2.fit(X_train2.fillna(-999), y_train)

    y_eval10 = pd.Series(rf.predict(X_eval.loc[10, :].fillna(-999)), index=X_eval.loc[10, :].index)
    y_eval11 = pd.Series(rf2.predict(X_eval2.loc[11, :].fillna(-999)), index=X_eval2.loc[11, :].index)
    y_eval = pd.concat([y_eval10, y_eval11], axis=0).to_frame('Log_Demanda')

    return y_eval, rf, rf2




if __name__ == '__main__':
    os.chdir(pts.working_dir())

    if not os.path.exists('Predictions'):
        os.makedirs('Predictions')
    if not os.path.exists('Predictions/release_v01'):
        os.makedirs('Predictions/release_v01')
    if not os.path.exists('Predictions/models_v01'):
        os.makedirs('Predictions/models_v01')

    start_time = datetime.datetime.now()

    states = town.State.unique()
    for i, state in enumerate(states):
        data = data_brush_for_model(state)
        print(state, 'read')
        y_eval, rf, rf2 = model_building(data)

        y_eval.to_csv('Predictions/release_v01/Prediction_%s_v01.csv' % state)

        with open('Predictions/models_v01/RF10week_%s.pkl' % state, 'wb') as model1, \
             open('Predictions/models_v01/RF11week_%s.pkl' % state, 'wb') as model2:
            pickle.dump(rf, model1, 2)
            pickle.dump(rf2, model2, 2)

        print('%s saved, %d to go, time:' % (state, len(states) - i - 1), datetime.datetime.now() - start_time)



