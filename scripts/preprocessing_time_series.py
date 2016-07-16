# -*- coding: utf-8 -*-
"""
Preprocessing for data

Created on Sun Jun 19 15:28:26 2016

@author: ilya
"""
from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
import re
import os
import unicodedata
import codecs

SPLITS = [['Producto_ID'], ['Producto_ID', 'Ruta_SAK'], ['Producto_ID', 'Cliente_ID', 'Agencia_ID']]

LAG_COLUMNS = [u'Venta_uni_hoy', u'Venta_hoy', u'Dev_uni_proxima',
       u'Dev_proxima', u'Demanda_uni_equil', u'Dev_proxima_by_uni', u'No_remains',
       u'Venta_hoy_by_uni', u'Ordered', u'Log_Demanda', u'Median_Producto_ID',
       u'LogMean_Producto_ID', u'Median_Producto_ID_Ruta_SAK',
       u'LogMean_Producto_ID_Ruta_SAK',
       u'Median_Producto_ID_Cliente_ID_Agencia_ID',
       u'LogMean_Producto_ID_Cliente_ID_Agencia_ID']

def working_dir():
    if 'ilya' in os.getcwd():
        directory = '/Users/ilya/Documents/Kaggle_Bimbo/'
    elif 'romul' in os.getcwd():
        os.chdir('/home/romul/kaggle/Bimbo/')
    else:
        directory = '/Users/margarita/Kaggle/Bimbo/'

    return directory


def select_random_rows(seed=0):
    np.random.seed(seed)
    n = 74180465  # number of records in file
    s = 1 * 10 ** 6  # desired sample size
    skip = np.sort(np.random.permutation(n)[:(n - s)])[1:]
    df_train = pd.read_csv(working_dir() + "train.csv", skiprows=skip)
    print('Data read')
    return df_train


def select_states(states):
    town = text_encoding(town_preproc())
    agencies = set([str(int(i)) for i in town.loc[town.State.isin(states)].index])

    with codecs.open(working_dir() + "train.csv", 'r', 'utf-8') as f:
        column_names = f.readline()[:-1].split(',')
        data = []
        for line in f.readlines():
            fields = line[:-1].split(',')
            if fields[1] in agencies:
                data.append(fields)

    df_train = pd.DataFrame(data, columns=column_names)
    print('Data read complete')

    for c in df_train:
        df_train[c] = pd.to_numeric(df_train[c], errors='coerce')

    print('To numeric encoded')

    return df_train


def town_preproc():
    town = pd.read_csv(working_dir() + 'town_state.csv', index_col=0)
    town.loc[town.Town == '2087 AG. TIZAYUCA', 'State'] = 'HIDALGO'  # because of error in state
    return town


def products_preproc():
    products = pd.read_csv(working_dir() + 'producto_tabla.csv', index_col=0)
    products['brand'] = products.NombreProducto.str.extract('^.+\s(\D+) \d+$', expand=False)
    w = products.NombreProducto.str.extract('(\d+)(Kg|g)', expand=True)
    products['weight'] = w[0].astype('float') * w[1].map({'Kg': 1000, 'g': 1})
    products['pieces'] = products.NombreProducto.str.extract('(\d+)p ', expand=False).astype('float')

    products['Has_choco'] = products['NombreProducto'].apply(lambda s: int(re.search("Choco", s) is not None))
    products['Has_vanilla'] = products['NombreProducto'].apply(lambda s: int(re.search("Va(i)?nilla", s) is not None))
    products['Has_multigrano'] = products['NombreProducto'].apply(lambda s: int(re.search("Multigrano", s) is not None))

    # products.drop('NombreProducto', axis=1, inplace=True)

    return products


def clients_preproc():
    clients = pd.read_csv(working_dir() + 'cliente_tabla.csv', index_col=0)
    clients.NombreCliente.replace(['SIN NOMBRE',
                                   'NO IDENTIFICADO'], np.nan, inplace=True)
    return clients


def volumes_preproc(data):
    if 'Dev_proxima' in data.columns:
        data['Dev_proxima_by_uni'] = data.Dev_proxima / data.Dev_uni_proxima
        data['No_remains'] = data.Dev_uni_proxima.apply(np.sign)
    if 'Venta_hoy' in data.columns:
        data['Venta_hoy_by_uni'] = data.Venta_hoy / data.Venta_uni_hoy
        data['Ordered'] = data.Venta_uni_hoy.apply(np.sign)
    if 'Demanda_uni_equil' in data.columns:
        data['Log_Demanda'] = data.Demanda_uni_equil.apply(np.log1p)
        data = medians_logmeans_compute(data)

        # data.drop('Demanda_uni_equil', axis=1, inplace=True)

    print('Variables added')
    return data


def medians_logmeans_compute(data, splits=SPLITS):
    for split in splits:
        group = data[split + ['Demanda_uni_equil', 'Log_Demanda']].groupby(split)
        aggregates = pd.DataFrame()
        median_name = '_'.join(['Median'] + split)
        aggregates[median_name] = group.Demanda_uni_equil.median()
        logmean_name = '_'.join(['LogMean'] + split)
        aggregates[logmean_name] = group.Log_Demanda.mean()
        data = data.merge(aggregates[[median_name, logmean_name]], how='left', left_on=split, right_index=True)
        print(logmean_name, 'and', median_name, 'added')

    return data


def text_encoding(data):
    for c in data.columns:
        if data[c].dtype.name == 'object':  # string
            d_enc = {}
            for s in data[c].unique():
                if type(s) == str or type(s) == unicode:
                    encoded = unicodedata.normalize('NFKD', s.decode('utf-8')).encode('ascii', 'ignore')
                    if s != encoded:
                        d_enc[s] = encoded

            data[c] = data[c].apply(lambda x: d_enc.get(x, x))
    print('Data endoded')
    return data


def lag_generation(df, n_lags=5):
    indexers = [u'Semana', u'Agencia_ID', u'Canal_ID',
                u'Ruta_SAK', u'Cliente_ID', u'Producto_ID']

    lag_columns = LAG_COLUMNS

    df_lagged = df.copy()

    for lag in range(1, n_lags + 1):
        df_lag = df.copy()
        df_lag.Semana = df_lag.Semana + lag
        df_lag = df_lag.set_index(indexers)[lag_columns]
        df_lag.rename(columns=dict([(value, '%s_%d' % (value, lag)) for value in df_lag.columns]), inplace=True)
        df_lagged = pd.merge(df_lagged, df_lag, 'left', left_on=indexers, right_index=True)
        print(lag, 'lag done')

    return df_lagged


def preproc_timeseries(states=None, train=True):
    if train:
        # df = select_states(states)
        filelist = os.listdir(working_dir() + 'States/')
        assert len(filelist) > 0, 'run data_split_by_state.py first'
        df_list = []
        for state in states:
            df_list.append(pd.read_csv(working_dir() + 'States/' + state))
        df = pd.concat(df_list, axis=0)
    else:
        df = pd.read_csv(working_dir() + "test.csv")
        print('Data read')

    town = town_preproc()
    products = products_preproc()
    clients = clients_preproc()

    data = pd.merge(df, town, 'left', left_on='Agencia_ID', right_index=True)
    data = pd.merge(data, products, 'left', left_on='Producto_ID', right_index=True)
    data = pd.merge(data, clients, 'left', left_on='Cliente_ID', right_index=True)
    print('Data merged')

    if train:
        data = volumes_preproc(data)
        data = lag_generation(data)

    # data.drop(['Agencia_ID', 'Producto_ID', 'Cliente_ID'], axis=1, inplace=True)

    data = text_encoding(data)

    return data


if __name__ == '__main__':
    states = ['SONORA']
    data_train = preproc_timeseries(states=states)
    data_train.to_csv('%strain_%s_timeseries.csv' % (working_dir(), '_'.join(states)), index=False)
    print('Saved')

    # data_test = preproc_timeseries(train=False)
    # data_test.to_csv(working_dir() + 'test_preprocessed.csv', index=False)
    # print('Saved')
