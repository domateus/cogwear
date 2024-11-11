# pyright can be annoying
from sklearn.preprocessing import MinMaxScaler
import neurokit2 as nk
from src.data.utils import Split
import numpy as np
import re
from matplotlib import pyplot as plt
from src.experiments.consts import ExperimentType
from src.experiments.experiment import Experiment
from src.signals.subject import Subject, Signal
from math import floor
import scipy.stats as stats
from random import randrange
from src.classifiers.svm import Svm
from src.classifiers.xgb import Xgb
from src.classifiers.knn import Knn

class EEGExperiment(Experiment):
    def __init__(self, classifier: str, type: ExperimentType, path: str):
        Experiment.__init__(self, 'eeg', classifier, type, path, 'muse', EEGSubject, 30)

    def run_once(self, hyperparameters, percentage_data=1.):
        if ExperimentType.END_TO_END == self.type or self.classifier == 'cnn':
            return super().run_once(hyperparameters, percentage_data)

        fold_no = randrange(10)
        logging_message = "Experiment for {} signal, classifier: {}, fold: {}".format(
            self.signal, self.classifier, fold_no)
        self.logger.info(logging_message)

        classifier = None
        if self.classifier == 'svm':
            classifier = Svm()
        if self.classifier == 'xgb':
            classifier = Xgb()
        if self.classifier == 'knn':
            classifier = Knn()

        loss = classifier.run_once(self, hyperparameters, fold_no, False)

        self.logger.info("Finished e" + logging_message[1:])

        return None, loss

    def shape(self):
        rounds = self.get_test_data()
        (x_test, y) = rounds[0]
        s = np.shape(x_test)
        print(f'shape: {s}')
        return (s[0], *s[-2:])

    def get_train_data(self, fold: Split, percentage_data=1.):
        x_train, y_train, x_test, y_test, x_val, y_val = super().get_train_data(fold, percentage_data)

        x_train = [*np.swapaxes(x_train, 0,1)]
        x_test = [*np.swapaxes(x_test, 0,1)]
        x_val = [*np.swapaxes(x_val, 0,1)]

        return x_train, y_train, x_test, y_test, x_val, y_val

    def get_test_data(self):
        rounds = super().get_test_data()

        for k, (x, y) in enumerate(rounds):
            print(f'was: {np.shape(x)}')
            x= [*np.swapaxes(x, 0,1)]
            print(f'x shape: {np.shape(x)}')
            rounds[k] = (x, y)

        (aha, _) =rounds[0]
        print(f'first round: {np.shape(aha)}')
        return rounds



class EEGSubject(Subject):
    def __init__(self, path, id, device, sensor, window_duration, experiment_type):
        self._eeg_cols = ['Alpha_TP9','Alpha_AF7','Alpha_AF8','Alpha_TP10','Beta_TP9','Beta_AF7','Beta_AF8','Beta_TP10','Gamma_TP9','Gamma_AF7','Gamma_AF8','Gamma_TP10','Theta_TP9','Theta_AF7','Theta_AF8','Theta_TP10','Delta_TP9','Delta_AF7','Delta_AF8','Delta_TP10']

        Subject.__init__(self, path=path, id=id, device="muse", sensor="eeg", window_duration=window_duration, experiment_type=experiment_type)

    def values(self):
        if ExperimentType.END_TO_END == self.experiment_type:
            return super().values()
        ws, ys = self.all_windows()
        result = []
        for w in ws:
            values = np.swapaxes(w, 0, 1) # shape: columns, signal at 4Hz
            row = []
            for col in values:
                for v in self.sts(col):
                    row.append(v)
            result.append(row)
                    
        result = MinMaxScaler().fit_transform(result)
        return result, ys

    def all_windows(self):
        # nk uses mne, underlying. Therefore data is converted into mne shape-like here
        mne_data = self._data[self._eeg_cols].transpose().values
        diss = nk.eeg_diss(mne_data)
        gfp = nk.eeg_gfp(mne_data, sampling_rate=self._x.sampling, method='l1', normalize=True)
        mne_data = np.append(mne_data, [diss, gfp], axis=0)
        mne_data[np.isnan(mne_data)] = 0
        signal = mne_data.transpose()
        ws = self._x.sampling * self.window_duration
        data = []
        ys = []
        for x in range(0, len(signal), ws):
            w = signal[x:x+ws]
            lines = np.shape(w)[0]
            y = [v for v in self._data['y'][x:x+ws]]
            if ws - lines > 0:
                w = np.concatenate([w, np.zeros(shape=(ws - lines, np.shape(w)[1]))])
            for _ in range(0, ws - lines):
                y.append(0)
            label = int(stats.mstats.mode(y).mode[0])
            ys.append(label)
            data.append(w)
        return data, ys

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
        _, ax = plt.subplots(5, figsize=(25, 30))
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

