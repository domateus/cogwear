from src.classifiers.tuning.hptuning import Tuner
from src.experiments.experiment import ExperimentType, Experiment
from src.signals.ppg import PPGExperiment
from src.signals.eda import EDAExperiment
from src.signals.eeg import EEGSubject
import os
from keras.api.backend import clear_session

classifiers = ['fcn', 'cnn', 'lstm', 'resnet']
path = os.getcwd()

for c in classifiers:
  print(c)
  # exp = PPGExperiment(classifier=c, type=ExperimentType.END_TO_END, path=path, device='samsung')
  # tuner = Tuner(exp)
  # tuner.tune(max_evals=40)
  # clear_session()
  # exp.losocv_run(tuner)
  # clear_session()
  # exp.test_best_models(tuner)

  # exp = PPGExperiment(classifier=c, type=ExperimentType.END_TO_END, path=path, device='empatica')
  # tuner = Tuner(exp)
  # tuner.tune(max_evals=40)
  # clear_session()
  # exp.losocv_run(tuner)
  # clear_session()
  # exp.test_best_models(tuner)

  exp = EDAExperiment(classifier=c, type=ExperimentType.END_TO_END, path=path)
  tuner = Tuner(exp)
  tuner.tune(max_evals=40)
  clear_session()
  # exp.losocv_run(tuner)
  # clear_session()
  # exp.test_best_models(tuner)

  # exp = Experiment(signal="eeg", classifier=c, type=ExperimentType.END_TO_END, path=path, device="muse", subject=EEGSubject)
  # tuner = Tuner(exp)
  # tuner.tune(max_evals=40)
  # clear_session()
  # exp.losocv_run(tuner)
  # clear_session()
  # exp.test_best_models(tuner)

# for c in classifiers:
#   print(c)
#   exp = PPGExperiment(classifier=c, type=ExperimentType.END_TO_END, path=path, device='empatica')
#   tuner = Tuner(exp)
#   tuner.tune(max_evals=40)
#   clear_session()
#   exp.losocv_run(tuner)
#   clear_session()
#   exp.test_best_models(tuner)

# for c in classifiers:
#   print(c)
#   exp = EDAExperiment(classifier=c, type=ExperimentType.END_TO_END, path=path)
#   tuner = Tuner(exp)
#   tuner.tune(max_evals=40)
#   clear_session()
#   exp.losocv_run(tuner)
#   clear_session()
#   exp.test_best_models(tuner)

# for c in classifiers:
#   exp = Experiment(signal="eeg", classifier=c, type=ExperimentType.END_TO_END, path=path, device="muse", subject=EEGSubject)
#   tuner = Tuner(exp)
#   tuner.tune(max_evals=40)
#   clear_session()
#   exp.losocv_run(tuner)
#   clear_session()
#   exp.test_best_models(tuner)