from src.classifiers.tuning.hptuning import Tuner
from src.experiments.experiment import ExperimentType, Experiment
from src.signals.eeg import EEGSubject
from src.signals.eda import EDAExperiment
import os
import gc 
import tensorflow as tf


gc.collect()
path = os.getcwd()

# exp = EDAExperiment(classifier='lstm', type=ExperimentType.END_TO_END, path=path)
# tuner = Tuner(exp)
# tuner.tune(max_evals=40)
fcn = Experiment(signal="eeg", classifier="lstm", type=ExperimentType.END_TO_END, path=path, device="muse", subject=EEGSubject)
# Tuner(fcn).tune(evals=1, max_evals=40)
tuner = Tuner(fcn)
# fcn.losocv_run(tuner)
fcn.test_best_models(tuner)
