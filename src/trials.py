from src.classifiers.tuning.hptuning import Tuner
from src.experiments.experiment import ExperimentType
from src.signals.ppg import PPGExperiment
from src.signals.eda import EDAExperiment
import os
from keras.api.backend import clear_session

classifiers = ['fcn', 'cnn', 'lstm', 'resnet']
path = os.getcwd()

for c in classifiers:
  print(c)
  exp = PPGExperiment(classifier=c, type=ExperimentType.END_TO_END, path=path, device='samsung')
  tuner = Tuner(exp)
  tuner.tune(max_evals=20)
  clear_session()
  tuner.tune(max_evals=40)

for c in classifiers:
  print(c)
  exp = PPGExperiment(classifier=c, type=ExperimentType.END_TO_END, path=path, device='empatica')
  tuner = Tuner(exp)
  tuner.tune(max_evals=20)
  clear_session()
  tuner.tune(max_evals=40)

for c in classifiers:
  print(c)
  exp = EDAExperiment(classifier=c, type=ExperimentType.END_TO_END, path=path)
  tuner = Tuner(exp)
  tuner.tune(max_evals=20)
  clear_session()
  tuner.tune(max_evals=40)
