# coding: utf-8
from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
import os, sys
import cPickle as pickle
import datetime
import xgboost as xgb

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)


# hook up all scripts
sys.path.append(os.path.abspath('../../scripts')) # if notebook in folder
import preprocessing_time_series as pts
from gridsearch import GridSearch

def load_data(state):
    data = pd.read_csv('Feature_releases/release_v02/train_%s.csv' % state)
    means_by_split = [x for x in data.columns if
                      'Mean' in x and not (x[-2] == '_' or x[-4:] == 'mean')]  # its mean, its not a lag

    data = data.drop(['Log_Dev_proxima', 'Log_Dev_uni_proxima',
                      'Log_Venta_hoy', 'Venta_uni_hoy', 'Venta_hoy',
                      'Dev_proxima', 'Demanda_uni_equil', 'Dev_proxima_by_uni', 'No_remains',
                      'Venta_hoy_by_uni'] + means_by_split, axis=1).set_index('Semana')
    return data

def grid_search_xgboost(param, tuning_param, X_train, X_test, y_train, y_test):
    xgb_model = xgb.XGBRegressor()
    xgb_model.set_params(**param)

    gs = GridSearch(xgb_model, tuning_param, verbose=1)
    gs.fit(X_train, X_test, y_train, y_test)

    return gs



if __name__ == '__main__':
    os.chdir(pts.working_dir())

    if not os.path.exists('ParameterTuning'):
        os.makedirs('ParameterTuning')
    if not os.path.exists('ParameterTuning/results_v02'):
        os.makedirs('ParameterTuning/results_v02')
    if not os.path.exists('ParameterTuning/input_parameters_v02'):
        os.makedirs('ParameterTuning/input_parameters_v02')

    # first argument is the name of the parameter file
    parameters_file = sys.argv[1]
    directory_path = 'ParameterTuning/results_v02/%s/'%parameters_file

    assert not os.path.exists(directory_path), '%s already exists' % directory_path
    os.makedirs(directory_path)

    # load parameters
    with open('ParameterTuning/input_parameters_v02/%s'%parameters_file, 'r') as f:
        state, parameters, tuning_parameters, weeks = eval(f.read())

    # stdout to logfile
    sys.stdout = Logger(directory_path + "logfile.txt")

    print('State:', state)
    print('Default parameters:', parameters)
    print('Tuning parameters:')
    for param in tuning_parameters:
        print(param)
    print()

    # load data, split train and validation
    data = load_data(state)
    X_train_week8 = data.loc[3:7, :].drop('Log_Demanda', axis=1)
    y_train = data.loc[3:7, :]['Log_Demanda']
    X_test_week8 = data.loc[8:9, :].drop('Log_Demanda', axis=1)
    y_test_week8 = data.loc[8:9, :]['Log_Demanda']
    del data
    X_train_week9 = X_train_week8[[col for col in X_train_week8.columns if '_1' != col[-2:]]]
    X_test_week9 = X_test_week8.loc[9, :][[col for col in X_test_week8.columns if '_1' != col[-2:]]]
    y_test_week9 = y_test_week8[9]
    y_test_week8 = y_test_week8[8]
    X_test_week8 = X_test_week8.loc[8, :]

    best_param_week8 = dict(parameters)
    best_param_week9 = dict(parameters)

    start_time = datetime.datetime.now()

    # in series tune different bach of parameters
    for stage, tuning_param in enumerate(tuning_parameters):
        print('Stage', stage)
        print('Tuning parameters:', tuning_param)

        if weeks == '8' or weeks == 'all':
            print('Week 8')

            gs_week8 = grid_search_xgboost(best_param_week8, tuning_param,
                                     X_train_week8, X_test_week8, y_train, y_test_week8)
            print('Best', gs_week8.best_params_)
            print('Train score: ', gs_week8.best_score_)

            # save GridSearch and all models in it
            with open(directory_path + 'gs_stage%d_week8'%stage, 'wb') as p:
                pickle.dump(gs_week8, p, 2)

            # update best_param for next stage
            for param in gs_week8.best_params_:
                best_param_week8[param] = gs_week8.best_params_[param]
            print('Best param week8 stage%d:'%stage, best_param_week8)
            print('Time:',  datetime.datetime.now() - start_time)

        # same for 9 week
        if weeks == '9' or weeks == 'all':
            print('Week 9')
            gs_week9 = grid_search_xgboost(best_param_week9, tuning_param,
                                     X_train_week9, X_test_week9, y_train, y_test_week9)
            print('Best', gs_week9.best_params_)
            print('Train score: ', gs_week9.best_score_)
            with open(directory_path + 'gs_stage%d_week9'%stage, 'wb') as p:
                pickle.dump(gs_week9, p, 2)

            for param in gs_week9.best_params_:
                best_param_week9[param] = gs_week9.best_params_[param]
            print('Best param week9 stage%d:'%stage, best_param_week9)
            print('Time:', datetime.datetime.now() - start_time)
        print()

    print('Grid Search end, time:', datetime.datetime.now() - start_time)
    print('Best parameters week8:', best_param_week8)
    print('Train score  week8:', gs_week8.best_score_)
    print('Best parameters week9:', best_param_week9)
    print('Train score  week9:', gs_week9.best_score_)













