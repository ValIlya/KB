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
        directory = '/home/romul/kaggle/Bimbo/'
    else:
        directory = '/Users/margarita/Kaggle/Bimbo/'

    return directory

def town_preproc():
    town = pd.read_csv(working_dir() + 'town_state.csv', index_col=0)
    town.loc[town.Town == '2087 AG. TIZAYUCA', 'State'] = 'HIDALGO'  # because of error in state
    return town


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
    return data


def preproc(states=None, train=True):
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
    data = pd.merge(df, town, 'left', left_on='Agencia_ID', right_index=True)

    if train:
        data = volumes_preproc(data)
        # data = lag_generation(data)

    data = text_encoding(data)

    return data


if __name__ == '__main__':

    out_dir = working_dir() + 'Feature_releases/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir = working_dir() + 'Feature_releases/release_v01/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    town = text_encoding(town_preproc())
    states = town.State.unique()
    for i, state in enumerate(states):
        data_train = preproc(states=[state])
        data_train.to_csv('%strain_%s.csv' % (out_dir, state), index=False)
        print('%s saved, %d to go' % (state, len(states) - i - 1))
