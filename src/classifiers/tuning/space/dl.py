from math import log10
from typing import Any
from src.exception import NoSuchClassifier
from src.classifiers.hyperparameters import Hyperparameters
from hyperopt import  hp
from src.classifiers.tuning.space.index import Space

class Deep(Space):
    def dict(self, hyperparameters: Any):
        return hp.dict()

    def yeehaw(self):
        result = {}
        result['cnn'] = {
            'lr_power': hp.randint("lr_power", 3, 7),
            'decay': hp.choice("decay", [.01, .001, .0001, .00001]),
            'reduce_lr_factor': hp.choice("reduce_lr_factor", [0.8, 0.5, 0.2, 0.1]),
            'batch_size': hp.choice("batch_size", [2, 4, 8, 16, 32]),
            'baseline_weight': hp.randint("baseline_weight", 2, 3),
            'filters_multiplier': hp.choice("filters_multiplier", [0.5, 1, 2]),
            'kernel_size_multiplier': hp.choice("kernel_size_multiplier", [0.5, 1, 2])
        }
        result["fcn"] = {
            'lr_power': hp.randint("lr_power", 3, 7),
            'decay': hp.choice("decay", [.01, .001, .0001, .00001]),
            'reduce_lr_factor': hp.choice("reduce_lr_factor", [0.8, 0.5, 0.2, 0.1]),
            'batch_size': hp.choice("batch_size", [2, 4, 8, 16, 32]),
            'baseline_weight': hp.randint("baseline_weight", 2, 3),
            'filters_multiplier': hp.choice("filters_multiplier", [0.5, 1, 2]),
            'kernel_size_multiplier': hp.choice("kernel_size_multiplier", [0.5, 1, 2])
        }
        result["lstm"] = {
            'lr_power': hp.randint("lr_power", 3, 7),
            'decay': hp.choice("decay", [.01, .001, .0001, .00001]),
            'reduce_lr_factor': hp.choice("reduce_lr_factor", [0.8, 0.5, 0.2, 0.1]),
            'batch_size': hp.choice("batch_size", [2, 4, 8, 16, 32]),
            'baseline_weight': hp.randint("baseline_weight", 2, 3),
            'filters_multiplier': hp.choice("filters_multiplier", [0.5, 1, 2]),
            'kernel_size_multiplier': hp.choice("kernel_size_multiplier", [0.5, 1, 2]),
            'lstm_units': hp.choice("lstm_units", [1, 2, 3])
        }
        result["resnet"] = {
            'lr_power': hp.randint("lr_power", 3, 7),
            'decay': hp.choice("decay", [.01, .001, .0001, .00001]),
            'reduce_lr_factor': hp.choice("reduce_lr_factor", [0.8, 0.5, 0.2, 0.1]),
            'batch_size': hp.choice("batch_size", [2, 4, 8, 16, 32]),
            'baseline_weight': hp.randint("baseline_weight", 2, 3),
            'filters': hp.choice("filters", [16, 32, 64]),
            'kernel_size_multiplier':hp.choice("kernel_size_multiplier", [1, 2, 4])
        }
        


        subspace1 = [hp.choice("filters", [16, 32, 64])]
        subspace2 = [hp.choice("kernel_size_multiplier", [1, 2, 4])]
        result["resnet"] = (optimizer_subspace, hp.choice("depth", [2, 3, 4]), subspace1, subspace2)

        return result[self.c]

    def get_search_space(self):
        result = {}
        optimizer_subspace = [hp.randint("lr_power", 3, 7), hp.choice("decay", [.01, .001, .0001, .00001]),
                              hp.choice("reduce_lr_factor", [0.8, 0.5, 0.2, 0.1]), hp.choice("batch_size", [2, 4, 8, 16, 32]), hp.randint("baseline_weight", 2, 3)]

        subspace1 = hp.choice(f"filters_multiplier", [0.5, 1, 2])
        subspace2 = hp.choice(f"kernel_size_multiplier", [0.5, 1, 2])
        space = (optimizer_subspace, subspace1, subspace2)
        result["cnn"] = space
        result["fcn"] = space

        result["lstm"] = (optimizer_subspace, subspace1, subspace2,hp.choice(f"lstm_units", [1, 2, 3]))

        subspace1 = [hp.choice("filters", [16, 32, 64])]
        subspace2 = [hp.choice("kernel_size_multiplier", [1, 2, 4])]
        result["resnet"] = (optimizer_subspace, hp.choice("depth", [2, 3, 4]), subspace1, subspace2)

        return result[self.c]

    def get_hyperparameters(self, x):
        lr, decay, reduce_lr_factor, batch_size, baseline_weight = x[0]

        if self.c in ["cnn", "fcn"]:
            return Hyperparameters(lr, decay, reduce_lr_factor,  batch_size, filters_multipliers=x[1], kernel_size_multiplier=x[2], baseline_weight=baseline_weight)
        if self.c == "lstm":
            return Hyperparameters(lr, decay, reduce_lr_factor, batch_size, filters_multipliers=x[1], kernel_size_multiplier=x[2], lstm_units=x[3])
        if self.c == "resnet":
            return Hyperparameters(lr, decay, reduce_lr_factor, batch_size, depth=x[1], filters=x[2][0], kernel_size_multiplier=x[3][0])
        raise NoSuchClassifier(self.c)

    def best_trial_hyperparameter(self, trials):
        trial = trials.best_trial['result']['x']
        filters_multipliers = trial['filters_multipliers']
        filters = trial[ 'filters' ]
        kernel_size_multiplier = trial[ 'kernel_size_multiplier' ]
        kernel_sizes = trial[ 'kernel_sizes' ]
        dense_outputs = trial[ 'dense_outputs' ]
        depth = trial[ 'depth' ]
        lstm_units = trial[ 'lstm_units' ]
        batch_size = trial[ 'batch_size' ]
        lr = trial[ 'lr' ]
        lr_power = log10(1 / lr)
        decay = trial[ 'decay' ]
        reduce_lr_factor = trial[ 'reduce_lr_factor' ]
        class_weights = trial[ 'class_weights' ]
        baseline_weight = class_weights[0]
        return Hyperparameters(lr_power, decay, reduce_lr_factor, batch_size, filters_multipliers, filters, kernel_size_multiplier, kernel_sizes, dense_outputs, depth, lstm_units, baseline_weight)

