from src.classifiers.tuning.index import Tuner
from src.experiments.experiment import Experiment
from src.experiments.consts import ExperimentType
from src.signals.ppg import PPGExperiment
from src.signals.eda import EDAExperiment
from src.signals.eeg import EEGSubject, EEGExperiment
import os
from keras.api.backend import clear_session

classifiers = ['cnn', 'fcn', 'resnet']
path = os.getcwd()

for c in classifiers:
  # exp = PPGExperiment(classifier=c, type=ExperimentType.END_TO_END, path=path, device='empatica')
  # tuner = Tuner(exp)
  # exp.test_best_models(tuner)
  #
  # exp = PPGExperiment(classifier=c, type=ExperimentType.END_TO_END, path=path, device='samsung')
  # tuner = Tuner(exp)
  # exp.test_best_models(tuner)
  #
  # exp = EDAExperiment(classifier=c, type=ExperimentType.END_TO_END, path=path)
  # tuner = Tuner(exp)
  # exp.test_best_models(tuner)

  exp = EEGExperiment(classifier=c, type=ExperimentType.END_TO_END, path=path)
  tuner = Tuner(exp)
  exp.test_best_models(tuner)

