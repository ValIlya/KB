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

def data_brush_for_model(state):
    data_train = pd.read_csv('Feature_releases/release_v02/train_%s.csv' % state)
    # only state from train set
    agencies = set(town.loc[town.State == state].index)

    data_test_state = data_test.loc[data_test.Agencia_ID.isin(agencies), :]
    data = pd.concat([data_train, data_test_state], axis=0)

    means_by_split = [x for x in data_train.columns if
                      'Mean' in x and not (x[-2] == '_' or x[-4:] == 'last')]  # its mean, its not a lag
    data = data_train.drop([u'Town', u'State', u'Venta_uni_hoy', u'Venta_hoy', u'Dev_uni_proxima',
                                      u'Dev_proxima', u'Demanda_uni_equil', u'Dev_proxima_by_uni', u'No_remains',
                                      u'Venta_hoy_by_uni', u'Ordered'] + means_by_split, axis=1).set_index(u'Semana')

    return data

def model_building(data, feat_week10, feat_week11):

    X_train = data.loc[~data.Log_Demanda.isnull(),:].drop('Log_Demanda', axis=1)
    y_train = data.loc[~data.Log_Demanda.isnull(),'Log_Demanda']
    X_eval = data.loc[data.Log_Demanda.isnull(),:].drop('Log_Demanda', axis=1)
    # for 11 week special
    X_train2 = X_train[[col for col in X_train.columns if '_1' != col[-2:]]]
    X_eval2 = X_eval[[col for col in X_eval.columns if '_1' != col[-2:]]]

    param1 = {
        'learning_rate': 0.3,
        'gamma': 1,
        'max_depth': 16,
        'min_child_weight': 18,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    param2 = {
        'learning_rate': 0.3,
        'gamma': 1,
        'max_depth': 16,
        'min_child_weight': 18,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    xgb_model = xgb.XGBRegressor()
    xgb_model.set_params(**param1)
    xgb_model.fit(X_train[feat_week10], y_train)

    xgb_model2 = xgb.XGBRegressor()
    xgb_model2.set_params(**param2)
    xgb_model2.fit(X_train2[feat_week11], y_train)

    y_eval10 = pd.Series(xgb_model.predict(X_eval.loc[10, :][feat_week10]), index=X_eval.loc[10, :].index)
    y_eval11 = pd.Series(xgb_model2.predict(X_eval2.loc[11, :][feat_week11]), index=X_eval2.loc[11, :].index)
    y_eval = pd.concat([y_eval10, y_eval11], axis=0).to_frame('Log_Demanda')

    return y_eval, xgb_model, xgb_model2




if __name__ == '__main__':
    os.chdir(pts.working_dir())

    if not os.path.exists('Predictions'):
        os.makedirs('Predictions')
    if not os.path.exists('Predictions/release_v02'):
        os.makedirs('Predictions/release_v02')
    if not os.path.exists('Predictions/models_v02'):
        os.makedirs('Predictions/models_v02')

    feat_imp_week10 = pd.Series.from_csv('Feature_releases/release_v02/feat_imp_week10_xgboost_release_v02.csv')
    feat_imp_week11 = pd.Series.from_csv('Feature_releases/release_v02/feat_imp_week11_xgboost_release_v02.csv')
    feat_week10 = list(feat_imp_week10[:50].index)
    feat_week11 = list(feat_imp_week11[:50].index)


    start_time = datetime.datetime.now()

    states = town.State.unique()[::-1]
    for i, state in enumerate(states):
        data = data_brush_for_model(state)
        print(state, 'read')
        y_eval, m1, m2 = model_building(data, feat_week10, feat_week11)

        y_eval.to_csv('Predictions/release_v02/Prediction_%s_v02.csv' % state)

        with open('Predictions/models_v02/xgboost_week_%s.pkl' % state, 'wb') as model1, \
             open('Predictions/models_v02/xgboost_week_%s.pkl' % state, 'wb') as model2:
            pickle.dump(m1, model1, 2)
            pickle.dump(m2, model2, 2)

        print('%s saved, %d to go, time:' % (state, len(states) - i - 1), datetime.datetime.now() - start_time)

    print('Final submit generating')
    state_files = ['Predictions/release_v02/Prediction_%s_v02.csv' % state for state in states]
    test_states = pd.concat([pd.read_csv(f, index_col=0) for f in state_files])
    test_data = pd.merge(data_test, test_states, how='left', right_index=True, left_index=True)
    pivot = test_data.groupby('Agencia_ID').Log_Demanda.apply(lambda ser: ser.isnull().mean())
    assert ((pivot>0).sum() == 0), 'Not all id were predicted'
    test_data['Demanda_uni_equil'] = test_data.Log_Demanda.apply(np.expm1)
    test_data.index.name = 'id'
    test_data[['Demanda_uni_equil']].to_csv('Predictions/release_v02/Prediction_v02.csv')
    print('Final submit saved')