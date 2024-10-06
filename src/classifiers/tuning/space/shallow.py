from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
import numpy as np

from src.classifiers.tuning.space.index import Space

class Shallow(Space):
    def dict(self, hyperparameters):
        return hyperparameters

    def get_hyperparameters(self, x):
        return x

    def best_trial_hyperparameter(self,trials):
        return trials.best_trial['result']['x']

    def get_search_space(self):
        result = {}
        result["xgb"] = {
            'max_depth': hp.choice("max_depth", np.arange(1,20,1,dtype=int)),
            'eta'      : hp.uniform("eta", 0, 1),
            'gamma'    : hp.uniform("gamma", 0, 100),
            'reg_alpha': hp.uniform("reg_alpha", 1e-8, 10),
            'reg_lambda' : hp.uniform("reg_lambda", 0,1),
            'colsample_bytree': hp.uniform("colsample_bytree", 0.5,1),
            'colsample_bynode': hp.uniform("colsample_bynode", 0.5,1), 
            'colsample_bylevel': hp.uniform("colsample_bylevel", 0.5,1),
            'n_estimators': hp.choice("n_estimators", np.arange(10,1000,10,dtype='int')),
            'min_child_weight' : hp.choice("min_child_weight", np.arange(1,10,1,dtype='int')),
            'max_delta_step' : hp.choice("max_delta_step", np.arange(1,10,1,dtype='int')),
            'subsample' : hp.uniform("subsample",0.1,1),
            'objective' : hp.choice('objective', ['binary:logistic', 'binary:hinge']) ,
            'eval_metric' : hp.choice('eval_metric', ['aucpr', 'logloss', 'rmse']),
            'seed' : 42
        }
        result["knn"] = {
            'n_neighbors': hp.randint('n_neighbors', 1, 10),
            'weights': hp.choice('weights', ['uniform', 'distance']),
            'algorithm': hp.choice('algorithm', ['ball_tree', 'kd_tree', 'brute']),
            'leaf_size': hp.randint('leaf_size', 2, 100),
            'p': hp.choice('p', [1, 2]),
            'n_jobs': -1
        }
        result["svm"] = {
            'C': hp.choice('C', [0.1, 0.01, 0.001, 0.0001, 1, 10, 100, 1000]),
            'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'degree': hp.randint('degree', 1, 5),
            'gamma': hp.choice('gamma', ['scale', 'auto']),
            'class_weight': hp.choice('class_weight', ['balanced', None])
        }
        return result[self.c]
