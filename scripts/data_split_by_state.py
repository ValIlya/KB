# -*- coding: utf-8 -*-
"""
Preprocessing for data

Created on Sun Jul 16

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
import datetime


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


def data_split_by_state():

    out_dir = working_dir() + 'States/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    town = text_encoding(town_preproc())
    town = town.reset_index()
    town['Agencia_ID'] = town['Agencia_ID'].astype(int).astype(str)
    agencies_dict = town.groupby('Agencia_ID').State.first().to_dict()

    states = town.State.unique()

    out_files = dict([(state, codecs.open(out_dir+state, 'w', 'utf-8')) for state in states])

    in_file = codecs.open(working_dir() + "train.csv", 'r', 'utf-8')

    start_time = datetime.datetime.now()

    # Put header in each file
    column_names = in_file.readline()
    for f in out_files.values():
        f.write(column_names)

    # Read in_file line by line and write it to particular file
    for i, line in enumerate(in_file):
        if i % 100000 == 0:
            print (i, 'lines read, time:', datetime.datetime.now() - start_time)

        fields = line[:-1].split(',')
        agency = fields[1]
        if agency in agencies_dict.keys():
            state = agencies_dict[agency]
            out_files[state].write(line)
        else:
            print (agency, 'not in agency list')

    print('Data read complete')

    for f in out_files.values():
        f.close()
    in_file.close()


if __name__ == '__main__':
    data_split_by_state()
