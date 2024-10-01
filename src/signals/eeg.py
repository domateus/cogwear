# pyright can be annoying
from src.data.utils import Split
import numpy as np
import re
from matplotlib import pyplot as plt
from src.experiments.consts import ExperimentType
from src.experiments.experiment import Experiment
from src.signals.subject import Subject, Signal
from math import floor
import scipy.stats as stats

class EEGExperiment(Experiment):
    def __init__(self, classifier: str, type: ExperimentType, path: str):
        Experiment.__init__(self, 'eeg', classifier, type, path, 'muse', EEGSubject, 30)

    def shape(self):
        x_test, _ = self.get_test_data()
        s = np.shape(x_test)
        return (s[0], *s[-2:])

    def get_train_data(self, fold: Split, percentage_data=1.):
        x_train, y_train, x_test, y_test, x_val, y_val = super().get_train_data(fold, percentage_data)

        x_train = [*np.swapaxes(x_train, 0,1)]
        x_test = [*np.swapaxes(x_test, 0,1)]
        x_val = [*np.swapaxes(x_val, 0,1)]

        return x_train, y_train, x_test, y_test, x_val, y_val

    def get_test_data(self):
        x, y = super().get_test_data()

        x= [*np.swapaxes(x, 0,1)]

        return x, y


class EEGSubject(Subject):
    def __init__(self, path, id, device, sensor, window_duration, experiment_type):
        self._eeg_cols = ['Alpha_TP9','Alpha_AF7','Alpha_AF8','Alpha_TP10','Beta_TP9','Beta_AF7','Beta_AF8','Beta_TP10','Gamma_TP9','Gamma_AF7','Gamma_AF8','Gamma_TP10','Theta_TP9','Theta_AF7','Theta_AF8','Theta_TP10','Delta_TP9','Delta_AF7','Delta_AF8','Delta_TP10']

        Subject.__init__(self, path=path, id=id, device="muse", sensor="eeg", window_duration=window_duration, experiment_type=experiment_type)

    def _get_signal(self):
        data = [self._data[col] for col in self._eeg_cols]
        return Signal(self.sensor, self.sensor, self.sampling, data)

    def _expand_dims_axis(self): # pyright: ignore[reportIncompatibleMethodOverride]
        return 2

    def _get_window(self, index=0, seconds=30, overlap_ratio=0.5): # pyright: ignore[reportIncompatibleMethodOverride]
        window_size = seconds * self._x.sampling
        initial_point = floor(index * window_size * (1 - overlap_ratio))
        y = int(stats.mstats.mode(self._y[initial_point:initial_point+window_size]).mode[0])
        x = [[it for it in signal[initial_point:initial_point+window_size]] for signal in self._x.data]
        for _ in range(0, window_size - len(x[0])):
            for s in x:
                s.append(0)
        return x, y

    def show(self, window, to_plot=[]):
        _, ax = plt.subplots(5, figsize=(25, 20))
        x = self.x()
        for i in range(0, np.shape(x)[1], 4):
            title = self._eeg_cols[i].split('_')[0]
            c1 = self._eeg_cols[i].split('_')[1]
            c2 = self._eeg_cols[i+1].split('_')[1]
            c3 = self._eeg_cols[i+2].split('_')[1]
            c4 = self._eeg_cols[i+3].split('_')[1]

            ax[i//4].plot(x[window][i], label=c1)
            ax[i//4].plot(x[window][i+1], label=c2)
            ax[i//4].plot(x[window][i+2], label=c3)
            ax[i//4].plot(x[window][i+3], label=c4)
            ax[i//4].legend()
            ax[i//4].title.set_text(title)

