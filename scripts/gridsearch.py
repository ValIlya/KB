# -*- coding: utf-8 -*-

import numpy as np
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

class GridSearch():
    
    def __init__(self, estimator, param_grid):
        self.estimator = estimator
        self.param_grid = param_grid

    
    def fit(self, X_train, X_test, y_train, y_test):
        self.best_score_ = np.inf
        self.grid_scores_ = []
        for param in ParameterGrid(self.param_grid):
            estimator = clone(self.estimator)
            estimator.set_params(**param)
            estimator.fit(X_train, y_train)
        
            # пока только rmse, потом если надо добавлю нормальный scoring
            score = np.sqrt(mean_squared_error(y_test, estimator.predict(X_test)))
            self.grid_scores_.append("score: %f, params: %s"%(score, str(param)))
            
            if score < self.best_score_:
                self.best_score_ = score
                self.best_params_ = param
                self.best_estimator_ = estimator
                
    def predict(X):
        return self.best_estimator_.predict(X)
