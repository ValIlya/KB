# -*- coding: utf-8 -*-
"""
Preprocessing for data

Created on Sun Jun 19 15:28:26 2016

@author: ilya
"""

import numpy as np
import pandas as pd
import re
import os
import unicodedata


def working_dir():
    if 'ilya' in os.getcwd():
        dir = '/Users/ilya/Documents/Kaggle_Bimbo/'
    elif 'romul' in os.getcwd():
        dir = '/home/romul/kaggle/Bimbo/'
    else:
        dir = '/Users/margarita/Kaggle/Bimbo/'
        
    return dir
    
def select_random_rows(seed = 0):
    np.random.seed(seed)
    n = 74180465 #number of records in file
    s = 1 * 10 ** 6 #desired sample size
    skip = np.sort(np.random.permutation(n)[:(n-s)])[1:]
    df_train = pd.read_csv(working_dir() + "train.csv", skiprows=skip)
    print('Data read')
    return df_train
    

def town_preproc():
    town = pd.read_csv(working_dir() + 'town_state.csv', index_col=0)
    town.loc[town.Town == '2087 AG. TIZAYUCA', 'State'] = 'HIDALGO' #because of error in state
    return town
    
def products_preproc():

    def get_brand(product):
        tokens = product.split(' ')[:-1]
        brand = ''
        for token in tokens[::-1]:
            if token.upper() == token:
                brand = token + ' ' + brand
            else:
                break
        return brand.strip()
    
    
    products = pd.read_csv(working_dir() + 'producto_tabla.csv', index_col=0)
    products['brand'] = products['NombreProducto'].apply(get_brand)
    
    products['weight'] = products['NombreProducto'].apply(lambda s: re.search("\\d+[Kg|g]", s))
    products.loc[~products['weight'].isnull(), 'weight'] = \
    products.loc[~products['weight'].isnull(), 'weight'].apply(lambda s: re.sub('g', '', re.sub('Kg', '000', s.group(0))))
    products['weight'] = pd.to_numeric(products['weight'], errors='coerce')

    products['pieces'] = products['NombreProducto'].apply(lambda s: re.search("\\d+p\\b", s))
    products.loc[~products['pieces'].isnull(), 'pieces'] = \
    products.loc[~products['pieces'].isnull(), 'pieces'].apply(lambda s: re.sub('\D', '', s.group(0)))
    products['pieces'] = pd.to_numeric(products['pieces'], errors='coerce')
    
    products['Has_choco'] = products['NombreProducto'].apply(lambda s: int(re.search("Choco", s) is not None))
    products['Has_vanilla'] = products['NombreProducto'].apply(lambda s: int(re.search("Va(i)?nilla", s) is not None))
    products['Has_multigrano'] = products['NombreProducto'].apply(lambda s: int(re.search("Multigrano", s) is not None))
    
    # products.drop('NombreProducto', axis=1, inplace=True)
    
    return products    
    

def clients_preproc():
    clients = pd.read_csv(working_dir() + 'cliente_tabla.csv', index_col = 0)
    clients.NombreCliente.replace(['SIN NOMBRE',
                                   'NO IDENTIFICADO'], np.nan, inplace=True)
    
    return clients
    
    
def variables_generation(data):
    if 'Dev_proxima' in data.columns: 
        data['Dev_proxima_by_uni'] = data.Dev_proxima / data.Dev_uni_proxima
        data['No_remains'] = data.Dev_uni_proxima.apply(np.sign)
    if 'Demanda_uni_equil' in data.columns:    
        data['Log_Demanda'] = (data.Demanda_uni_equil+1).apply(np.log)
    
    print('Variables added')
    return data
    
    
def text_encoding(data):
    
    for c in data.columns:
        if data[c].dtype.name == 'object': #string
            d_enc = {}
            for s in data[c].unique():
                if type(s) == str or type(s) == unicode:
                    encoded  = unicodedata.normalize('NFKD', s.decode('utf-8')).encode('ascii','ignore')
                    if s != encoded:
                        d_enc[s] = encoded
        
            data[c] = data[c].apply(lambda x: d_enc.get(x, x))
    print('Data endoded')
    return data
    
    
def preproc(train = True):
    if train:
        df = select_random_rows()
    else:
        df = pd.read_csv(working_dir() + "test.csv")
        print('Data read')
    
    town = town_preproc()
    products = products_preproc()
    clients = clients_preproc()
    
    data = pd.merge(df, town, 'left', left_on = 'Agencia_ID', right_index = True)
    data = pd.merge(data, products, 'left', left_on = 'Producto_ID', right_index = True)
    data = pd.merge(data, clients, 'left', left_on = 'Cliente_ID', right_index = True)
    print('Data merged')
    # data.drop(['Agencia_ID', 'Producto_ID', 'Cliente_ID'], axis=1, inplace=True)

    
    data = text_encoding(data)
    
    data = variables_generation(data)
    
    return data
    
    
    
if __name__ == '__main__':

    data_train = preproc()
    data_train.to_csv(working_dir() + 'train_sample.csv', index=False)
    print('Saved')
    data_test = preproc(train=False)
    data_test.to_csv(working_dir() + 'test_preprocessed.csv', index=False)
    print('Saved')
