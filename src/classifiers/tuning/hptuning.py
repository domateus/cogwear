from math import log10
import os
import pickle

import tensorflow as tf

from src.experiments.experiment import Experiment
from src.exception import NoSuchClassifier
from src.classifiers.hyperparameters import Hyperparameters
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from keras.api.backend import clear_session

def get_search_space(classifier_name):
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

    return result[classifier_name]

def get_hyperparameters(classifier, x):
    lr, decay, reduce_lr_factor, batch_size, baseline_weight = x[0]

    if classifier in ["cnn", "fcn"]:
        return Hyperparameters(lr, decay, reduce_lr_factor,  batch_size, filters_multipliers=x[1], kernel_size_multiplier=x[2], baseline_weight=baseline_weight)
    if classifier == "lstm":
        return Hyperparameters(lr, decay, reduce_lr_factor, batch_size, filters_multipliers=x[1], kernel_size_multiplier=x[2], lstm_units=x[3])
    if classifier == "resnet":
        return Hyperparameters(lr, decay, reduce_lr_factor, batch_size, depth=x[1], filters=x[2][0], kernel_size_multiplier=x[3][0])
    raise NoSuchClassifier(classifier)

def best_trial_hyperparameter(trials):
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


def best_trial_hyperparameter_2(classifier, trials):
    best = {}
    for key in trials.best_trial['misc']['vals']:
      best[key] = trials.best_trial['misc']['vals'][key][0]
    return get_hyperparameters(classifier, space_eval(get_search_space(classifier), best))


class Tuner():
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.logger = experiment.logger
        self.trial_path = experiment.trials_path
        os.makedirs(self.trial_path, exist_ok=True)

    def tune(self, evals, max_evals):
        trials = self.load_trials()
        start = len(trials)
        finish = min(start + evals, max_evals)
        self.logger.info(f"trials length: {len(trials)}")
        space = get_search_space(self.experiment.classifier)
        for i in range(start, finish):
            fmin(fn=lambda x: self._objective(x, i),
                 space=space,
                 algo=tpe.suggest,
                 max_evals=i + 1,
                 trials=trials)

            # Moved into the loop due to preemptible gpu
            self.save_trials(trials)
        return trials

    def _objective(self, x, iteration):
        self.logger.info(f"Running objective for {self.experiment.classifier} at {iteration} iteration")
        hp = get_hyperparameters(self.experiment.classifier, x)
        with tf.device('/device:GPU:0'):
            clear_session()
            _, loss = self.experiment.run_once(hp, percentage_data=.2)
            clear_session()

        return {"status": STATUS_OK,
                "x": hp.dict(),
                "loss": loss}
            
    def best_hyperparameters(self):
        return best_trial_hyperparameter(self.load_trials())

    def load_trials(self):
        trials_filename = os.path.join(self.trial_path, self.experiment.classifier + ".pkl")

        try:
            if os.path.exists(trials_filename):
                with open(trials_filename, "rb") as f:
                    self.logger.info(f"loaded existing trials")
                    return pickle.load(f)
        except EOFError:
            pass

        trials = Trials()
        self.save_trials(trials)
        return trials

    def save_trials(self, trials):
        with open(os.path.join(self.trial_path, self.experiment.classifier + ".pkl"), "wb") as f:
            pickle.dump(trials, f)
