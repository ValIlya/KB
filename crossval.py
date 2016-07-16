# -*- coding: utf-8 -*-
"""
Cross-validation

@author: ilya
"""

import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold

def _create_combinations(columns):
    assert(isinstance(columns, pd.DataFrame), 'columns must be dataframe')
    df = columns.copy()
    for c in df:
        df[c] = df[c].astype(str)
    strats = df.apply(lambda row: '_'.join(row), axis=1).astype('category')
    return strats


def crossvalidation(columns = None, data_len=None, random_state = 0):
    assert((columns is not None) or (data_len is not None), 'give columns as DataFrame or data length')
    if columns is None:
        cv5fold = KFold(data_len, n_folds=5, random_state=random_state)
        strats = None
    else:
        strats = _create_combinations(columns)
        cv5fold = StratifiedKFold(strats, n_folds=5, random_state=random_state)
    return cv5fold, strats
