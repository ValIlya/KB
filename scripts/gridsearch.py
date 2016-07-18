# -*- coding: utf-8 -*-

import numpy as np
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

class GridSearch():
    
    def __init__(self, estimator, param_grid, verbose=0):
        self.estimator = clone(estimator)
        self.param_grid = param_grid
        self.verbose = verbose

    
    def fit(self, X_train, X_test, y_train, y_test):
        self.best_score_ = np.inf
        self.grid_scores_ = []

        parameter_grid = ParameterGrid(self.param_grid)

        if self.verbose:
            print ("Fitting %d candidates"%len(parameter_grid))

        for i, param in enumerate(parameter_grid):
            estimator = clone(self.estimator)
            estimator.set_params(**param)
            estimator.fit(X_train, y_train)
        
            # пока только rmse, потом если надо добавлю нормальный scoring
            test_score = np.sqrt(mean_squared_error(y_test, estimator.predict(X_test)))
            train_score = np.sqrt(mean_squared_error(y_train, estimator.predict(X_train)))
            self.grid_scores_.append("test score: %f, train score: %f, params: %s"
                                        %(test_score, train_score, str(param)))

            if self.verbose:
                print ("%d %s"%(i ,self.grid_scores_[-1]))
            
            if test_score < self.best_score_:
                self.best_score_ = test_score
                self.best_params_ = param
                self.best_estimator_ = estimator
                
    def predict(self, X):
        return self.best_estimator_.predict(X)
