from src.classifiers.tuning.hptuning import Tuner
from src.experiments.experiment import Experiment
from src.experiments.consts import ExperimentType
from src.signals.ppg import PPGExperiment
from src.signals.eda import EDAExperiment
from src.signals.eeg import EEGSubject
import os
from keras.api.backend import clear_session

classifiers = ['lstm']
path = os.getcwd()

for c in classifiers:
  print(c)
  # clear_session()
  # exp = PPGExperiment(classifier=c, type=ExperimentType.END_TO_END, path=path, device='empatica')
  # tuner = Tuner(exp)
  # tuner.tune(evals=40, max_evals=40)
  # exp.losocv_run(tuner)
  # clear_session()
  # exp.test_best_models(tuner)

  # exp = EDAExperiment(classifier=c, type=ExperimentType.END_TO_END, path=path)
  # tuner = Tuner(exp)
  # tuner.tune(evals=40, max_evals=40)
  # clear_session()
  # exp.losocv_run(tuner)
  # clear_session()
  # exp.test_best_models(tuner)

  exp = Experiment(signal="eeg", classifier=c, type=ExperimentType.END_TO_END, path=path, device="muse", subject=EEGSubject)
  tuner = Tuner(exp)
  # tuner.tune(evals=5, max_evals=40)
  # clear_session()
  # exp.losocv_run(tuner)
  # exp.losocv_run_once(tuner, 8)
  # clear_session()
  # exp.losocv_run_once(tuner, 9)
  # clear_session()
  exp.test_best_models(tuner)

