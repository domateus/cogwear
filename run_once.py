from src.classifiers.tuning.hptuning import Tuner
from src.experiments.experiment import ExperimentType, Experiment
from src.signals.eeg import EEGSubject
import os

path = os.getcwd()
fcn = Experiment(signal="eeg", classifier="fcn", type=ExperimentType.END_TO_END, path=path, device="muse", subject=EEGSubject)
# Tuner(fcn).tune(max_evals=40)
tuner = Tuner(fcn)
fcn.losocv_run(tuner)
