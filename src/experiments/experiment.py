import pandas as pd
from math import ceil
from random import randrange
from src.data.utils import TEST_SUBJECT_IDS, Split, create_classifier,losocv_splits
from abc import ABC
from src.data.utils import SUBJECTS_IDS
from src.signals.subject import Subject
import os
import numpy as np
from src.logger import  logger
from enum import Enum
from src.classifiers.hyperparameters import Hyperparameters
from keras.api.backend import clear_session
from src.experiments.consts import ExperimentType


class Experiment(ABC):
    def __init__(self, signal: str, classifier: str, type: ExperimentType, path: str, device:str, subject=Subject, window_duration=30):
        self._tuning_iteration = 0
        self.logger = logger
        self.signal = signal
        self.type = type
        self.classifier = classifier
        self.path = path
        self.data_path = os.path.join(path, 'survey_gamification')
        subjects = SUBJECTS_IDS
        test_subjects = TEST_SUBJECT_IDS
        self.device = device
        self.window_duration = window_duration
        results_folder_name = f'results_{window_duration}'
        self.trials_path = os.path.join(path, results_folder_name, self.type.name, f"{self.device}_{self.signal}", "trials", self.classifier)
        self.losocv_path = os.path.join(path, results_folder_name, self.type.name, f"{self.device}_{self.signal}", "losocv", self.classifier)
        self.test_path = os.path.join(path, results_folder_name, self.type.name, f"{self.device}_{self.signal}", "test", self.classifier)
        self.test_metrics_path =  os.path.join(path, results_folder_name, self.type.name, f"{self.device}_{self.signal}", "test")
        self.analysis_path = os.path.join(path, results_folder_name, self.type.name, f"{self.device}_{self.signal}", "analysis")
        self.subjects = [subject(path=self.data_path, id=f"{id}", device=device, sensor=signal, window_duration=window_duration, experiment_type=type) for id in subjects]
        self.test_subjects = [subject(path=self.data_path, id=f"{id}", device=device, sensor=signal, window_duration=window_duration, experiment_type=type) for id in test_subjects]
        self.splits: list[Split] = []
        for split in losocv_splits():
            self.splits.append(split.into(self.subjects))

    def shape(self):
        x_test, y_test = self.get_test_data()
        s = np.shape(x_test)
        return s[1:]


    def get_train_data(self, fold: Split, percentage_data=1.):
        x_train, y_train, x_test, y_test, x_val, y_val = fold.x_train(), fold.y_train(), fold.x_test(), fold.y_test(), fold.x_val(), fold.y_val()
        x_train, y_train, x_test, y_test, x_val, y_val = self._partial(x_train, percentage_data), self._partial(y_train, percentage_data), self._partial(x_test, percentage_data), self._partial(y_test, percentage_data), self._partial(x_val, percentage_data),self._partial(y_val, percentage_data) 
        return [*x_train], y_train, [*x_test], y_test, [*x_val], y_val

    def get_test_data(self):
        x = [*np.concatenate([s.x() for s in self.test_subjects])]
        y = np.concatenate([s.y() for s in self.test_subjects])
        return x, y


    def run_once(self, hyperparameters: Hyperparameters, percentage_data=1.):
        fold_no = randrange(10)
        logging_message = "Experiment for {} signal, classifier: {}, fold: {}".format(
            self.signal, self.classifier, fold_no)
        self.logger.info(logging_message)

        fold = self.splits[fold_no]
        x_train, y_train, x_test, y_test, x_val, y_val = self.get_train_data(fold, percentage_data)

        classifier = create_classifier(classifier_name=self.classifier, output_directory=self.trials_path, input_shape=self.shape(), hyperparameters=hyperparameters, fold=-1)

        metrics, loss = classifier.fit(x_train, y_train, x_val, y_val, y_test, x_test=x_test, nb_epochs=hyperparameters.epochs,
                       batch_size=hyperparameters.batch_size)

        self.logger.info("Finished e" + logging_message[1:])

        return metrics, loss

    def losocv_run_once(self, tuner, fold_index):
        # Generate at least once the hyperparmeters
        if len(tuner.load_trials()) == 0:
            tuner.tune(1)
        hyperparameters = tuner.best_hyperparameters()
        fold = self.splits[fold_index]
        clear_session()
        x_train, y_train, x_test, y_test, x_val, y_val = self.get_train_data(fold)

        classifier = create_classifier(classifier_name=self.classifier, output_directory=self.losocv_path, input_shape=self.shape(), hyperparameters=hyperparameters, fold=fold.id)

        metrics, loss = classifier.fit(x_train, y_train, x_val, y_val, y_test, x_test=x_test, nb_epochs=hyperparameters.epochs,
                       batch_size=hyperparameters.batch_size)

        self.logger.info(f"Fold: {fold.id} => loss: {loss}")

    def losocv_run(self, tuner):
        for i in range (0, len(self.splits)):
            self.losocv_run_once(tuner, i);

    def test_best_models(self, tuner):
        os.makedirs(self.test_path, exist_ok=True)
        x, y = self.get_test_data()
        if len(tuner.load_trials()) == 0:
            tuner.tune(1)
        hyperparameters = tuner.best_hyperparameters()
        best_models = [f for f in os.listdir(self.losocv_path) if 'best_model.weights' in f]
        results = {}
        for model in best_models:
            fold_id = model[0:2]
            classifier = create_classifier(classifier_name=self.classifier, output_directory=self.test_path, input_shape=self.shape(), hyperparameters=hyperparameters, fold=fold_id)
            print(f"model path: {os.path.join(self.losocv_path, model)}")

            metrics = classifier.predict(x, y, os.path.join(self.losocv_path, model))
            results[fold_id] = metrics
        return results

    def _partial(self, data, percentage):
        size = len(data) - 1
        total = ceil(size * percentage)
        if total % 2 == 1:
            total += 1
        result = np.array([*data[0:int(total/2)], *data[ceil(size / 2):ceil(size / 2) + int(total/2) - 1]])
        return result

