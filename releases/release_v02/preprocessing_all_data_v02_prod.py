# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
import os
import unicodedata

# State included in every split by design
SPLITS = [['Producto_ID', 'Cliente_ID', 'Ruta_SAK'],
          ['Producto_ID', 'Cliente_ID', 'Agencia_ID']]

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
        data['Log_Dev_proxima'] = data.Dev_proxima.apply(np.sign)
        data['Log_Dev_uni_proxima'] = data.Dev_uni_proxima.apply(np.sign)
        data['No_remains'] = data.Dev_uni_proxima.apply(np.sign)
        data.drop('Dev_uni_proxima',axis=1,inplace=True)
    if 'Venta_hoy' in data.columns:
        data['Venta_hoy_by_uni'] = data.Venta_hoy / data.Venta_uni_hoy
        data['Log_Venta_hoy'] = data.Venta_hoy.apply(np.sign)
        # data['Log_Venta_uni_hoy'] = data.Venta_uni_hoy.apply(np.sign)
        # data['Ordered'] = data.Venta_uni_hoy.apply(np.sign)
    if 'Demanda_uni_equil' in data.columns:
        data['Log_Demanda'] = data.Demanda_uni_equil.apply(np.log1p)
        data = logmeans_compute(data)
    return data


def logmeans_compute(data, splits=SPLITS):
    for split in splits:
        raw_cols = ['Log_Demanda', 'Log_Venta_uni_hoy', 'No_remains', 'Ordered']
        raw_cols = [x for x in raw_cols if x in data.columns]
        group = data[split + raw_cols].groupby(split)
        aggregates = pd.DataFrame()
        for field in raw_cols:
            name = '_'.join(['%s_Mean' % field] + split)
            aggregates[name] = group[field].mean()

        data = data.merge(aggregates, how='left', left_on=split, right_index=True)

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

def products_preproc():
    products = pd.read_csv(working_dir() + 'producto_tabla.csv', index_col=0)
    products['brand'] = products.NombreProducto.str.extract('^.+\s(\D+) \d+$', expand=False)
    products['brand'] = products['brand'].astype(str).astype('category').cat.codes # integers from categories
    w = products.NombreProducto.str.extract('(\d+)(Kg|g)', expand=True)
    products['weight'] = w[0].astype('float') * w[1].map({'Kg': 1000, 'g': 1})
    products['pieces'] = products.NombreProducto.str.extract('(\d+)p ', expand=False).astype('float')

    # products['Has_choco'] = products['NombreProducto'].apply(lambda s: int(re.search("Choco", s) is not None))
    # products['Has_vanilla'] = products['NombreProducto'].apply(lambda s: int(re.search("Va(i)?nilla", s) is not None))
    # products['Has_multigrano'] = products['NombreProducto'].apply(lambda s: int(re.search("Multigrano", s) is not None))

    # products.drop('NombreProducto', axis=1, inplace=True)

    products = products[['brand', 'weight', 'pieces']]

    return products

def lag_generation(df, n_lags=3, widths = [3, 4]):
    indexers = [u'Semana', u'Agencia_ID', u'Canal_ID',
                u'Ruta_SAK', u'Cliente_ID', u'Producto_ID']

    #only volumes are lagged
    lag_columns = [x for x in df.columns if ('Demanda' in x) or ('No_remains' in x) or
                   ('Venta' in x) or ('Dev_proxima' in x) or ('Ordered' in x)]

    df_lagged = df.copy()

    for lag in range(1, n_lags + 1):
        df_lag = df.copy()
        df_lag.Semana = df_lag.Semana + lag
        df_lag = df_lag.set_index(indexers)[lag_columns]
        df_lag.rename(columns=dict([(value, '%s_%d' % (value, lag)) for value in df_lag.columns]), inplace=True)
        df_lagged = pd.merge(df_lagged, df_lag, 'left', left_on=indexers, right_index=True)
        print(lag, 'lag done')

    df_lagged = wide_lag_generation(df_lagged, widths, lag_columns)

    return df_lagged


def wide_lag_generation(df, width_range, lag_columns):
    # means trought several weeks

    indexers = [u'Semana', u'Agencia_ID', u'Canal_ID',
                    u'Ruta_SAK', u'Cliente_ID', u'Producto_ID']

    df_lagged = df
    #first is a necessary base part - previous week
    df_lag_part = df.copy()
    df_lag_part.Semana = df_lag_part.Semana + 1
    df_lag = df_lag_part[indexers+lag_columns]

    for lag_width in range(2, width_range[1] + 1):
        df_lag_part = df.copy()
        df_lag_part.Semana = df_lag_part.Semana + lag_width
        df_lag_part = df_lag_part.loc[df_lag_part.Semana.isin(set(range(3, 12))), indexers+lag_columns]
        # every time add shifted semana to df_lag, then aggregate
        df_lag = pd.concat([df_lag, df_lag_part], axis=0)
        if lag_width >= width_range[0]:
            aggregates = df_lag.groupby(indexers).mean().rename(columns=dict([(value, '%s_%dmean' % (value, lag_width)) for value in df_lag.columns]))
            df_lagged = pd.merge(df_lagged, aggregates, 'left', left_on=indexers, right_index=True)
            print('%d-wide lag done' % lag_width)

    return df_lagged

def preproc(states=None, train=True):

    # df = select_states(states)
    filelist = os.listdir(working_dir() + 'States/')
    assert len(filelist) > 0, 'run data_split_by_state.py first'
    df_list = []
    for state in states:
        df_list.append(pd.read_csv(working_dir() + 'States/' + state))
    df = pd.concat(df_list, axis=0)

    # town = town_preproc()
    products = products_preproc()
    # data_train = pd.merge(df, town, 'left', left_on='Agencia_ID', right_index=True)
    data_train = pd.merge(df, products, 'left', left_on='Producto_ID', right_index=True)

    data_train = volumes_preproc(data_train)

    agencies = set(town.loc[town.State == state].index)
    data_test = pd.read_csv(working_dir() + 'test.csv', index_col=0)

    data_test_state = data_test.loc[data_test.Agencia_ID.isin(agencies), :]
    data = pd.concat([data_train, data_test_state], axis=0)

    # split data in parts by Producto_ID and calculate lags for each part independantly

    n_parts = int(data.shape[0] / (6 * 10 ** 5))

    products = data.Producto_ID.value_counts().sort_index()
    products_parts = []
    prod_set = set()
    cumulative_count = 0
    count_threshold = int(data.shape[0] / n_parts)
    for prod, count in products.iteritems():
        cumulative_count += count
        prod_set.add(prod)
        if cumulative_count >= count_threshold:
            products_parts.append(prod_set)
            prod_set = set()
            cumulative_count = 0
    products_parts.append(prod_set)
    print('file was splitted into', n_parts, 'parts by product')
    # Generating lags
    data_parts = [data.loc[data.Producto_ID.isin(prod_set), :] for prod_set in products_parts]
    for i, data_part in enumerate(data_parts):
        data_parts[i] = lag_generation(data_part)
    data = pd.concat(data_parts)

    data = text_encoding(data)

    return data


if __name__ == '__main__':

    out_dir = working_dir() + 'Feature_releases/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir = working_dir() + 'Feature_releases/release_v02/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    town = text_encoding(town_preproc())
    states = town.State.unique()[:3]
    for i, state in enumerate(states):
        data_train = preproc(states=[state])
        data_train.to_csv('%strain_%s.csv' % (out_dir, state), index=False)
        print('%s saved, %d to go' % (state, len(states) - i - 1))
