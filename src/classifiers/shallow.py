from abc import ABC, abstractmethod
from sklearn.model_selection import KFold, cross_val_score
import time
import os
import pickle
from random import randrange
from src.logger import log_predicions
import numpy as np
from src.experiments.experiment import Experiment
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, log_loss
from typing import Any

class ShallowClassifier(ABC):
    def __init__(self):
        self.preds = {}

    def test_values(self, experiment: Experiment):
      x_test = np.concatenate([s.x() for s in experiment.test_subjects])
      y_test = np.concatenate([s.y() for s in experiment.test_subjects])

      cols = x_test.shape[-1:][0]
      x_test = np.reshape(x_test, (1, -1, cols))[0]
      y_test = np.reshape(y_test, (1, -1))[0]
      return x_test, y_test

    @abstractmethod
    def build_model(self, hyperparameters) -> Any:
        pass

    def save_model(self, model, experiment: Experiment, fold):
        with open(os.path.join(experiment.losocv_path, f'{experiment.classifier}_{fold}.pkl'),'wb') as f:
            pickle.dump(model, f) 

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def run_once(self, experiment: Experiment, hyperparameters, fold_to_use=-1, save_model=True):
        os.makedirs(experiment.losocv_path, exist_ok=True)
        fold_index = fold_to_use if fold_to_use != -1 else randrange(len(experiment.splits))
        fold = experiment.splits[fold_index]
        X = np.concatenate([fold.x_train(), fold.x_val()])
        y = np.concatenate([fold.y_train(), fold.y_val()])

        cols = X.shape[-1:][0]
        X = np.reshape(X, (1, -1, cols))[0]
        y = np.reshape(y, (1, -1))[0]
        x_test = np.reshape(fold.x_test(), (1, -1, cols))[0]
        y_test = np.reshape( fold.y_test(), (1, -1) )[0]

        clf = self.build_model(hyperparameters)
        clf.fit(X, y)
        y_pred = clf.predict(x_test)
        loss = log_loss(y_test, y_pred)

        if save_model:
            self.save_model(clf, experiment, fold.id)
        return loss

    def cv_train(self, experiment: Experiment, hyperparameters):
      for i in range(0, len(experiment.splits)):
        self.run_once(experiment, hyperparameters, fold_to_use=i, save_model=True)

    def predict(self, experiment: Experiment):
        os.makedirs(experiment.test_path, exist_ok=True)
        x_test, y_test = self.test_values(experiment)
        classifiers = [f for f in os.listdir(experiment.losocv_path) if 'pkl' in f]
        for filename in classifiers:
            f = os.path.join(experiment.losocv_path, filename)
            clf = self.load_model(f)

            start_time = time.time()
            y_pred = clf.predict(x_test)
            duration = time.time() - start_time

            fold_id = filename.split('.')[0].split('_')[1]
            log_predicions(experiment.test_path, y_pred, y_test, duration, fold_id)

    def show_results(self):
      for x in self.preds:
        print(f'Subject {x}')
        y_true = self.preds[x]['y_true']
        y_pred = self.preds[x]['y_pred']
        print(classification_report(y_pred,y_true))

        plt.figure(figsize=(16, 8))
        plt.plot(y_true, 'b-', y_pred, 'r.')
        plt.show()
