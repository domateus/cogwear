from math import floor
from abc import ABC
import numpy as np
import heartpy as hp
from matplotlib import pyplot as plt
import itertools as it
import scipy.stats as stats
import scipy.signal as sig
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.classifiers.classifier import Classifier
from src.data.utils import Split
from src.signals.subject import Subject
from src.signals.utils import filter_signal
from src.signals.signal import Signal
from src.experiments.consts import  ExperimentType
from src.experiments.experiment import Experiment

class PPGExperiment(Experiment):
    def __init__(self, path: str, type: ExperimentType, classifier: str, device="samsung", window_duration=30):
        Experiment.__init__(self, signal="ppg", classifier=classifier,type=type, device= device, path=path, window_duration=window_duration, subject=PPGSubject)
        self.device = device
        self.path = path

    def shape(self):
        if ExperimentType.FEATURE_ENGINEERING:
            x_test, y_test = self.get_test_data()
            s = np.shape(x_test)
            # cols, window_size, 1
            return (s[0], *s[-2:])
        else:
            return super().shape()

    def get_test_data(self):
        x, y = super().get_test_data()
        if ExperimentType.FEATURE_ENGINEERING == self.type:
            x = np.array(x)
            cols = x.shape[-1:][0]
            x = [*x.reshape((cols, -1, self.window_duration, 1))]
            new_y= []
            for w in y:
                label = int(stats.mstats.mode(w).mode[0])
                new_y.append([ label ])
            y = np.array(new_y)
        return x, y

    def get_train_data(self, fold: Split, percentage_data=1.):
        x_train, y_train, x_test, y_test, x_val, y_val = super().get_train_data(fold, percentage_data)
        if ExperimentType.FEATURE_ENGINEERING == self.type:
            cols = np.shape(x_val)[-1:][0]
            # they need to be a normal list for keras to work
            x_train = [*np.reshape(x_train,(cols,-1, self.window_duration, 1))]
            x_test = [*np.reshape(x_test,(cols,-1, self.window_duration, 1))]
            x_val = [*np.reshape(x_val,(cols,-1, self.window_duration, 1))]
            new_y_train = []
            for w in y_train:
                label = int(stats.mstats.mode(w).mode[0])
                new_y_train.append([ label ])
            y_train = np.array(new_y_train)

            new_y_test = []
            for w in y_test:
                label = int(stats.mstats.mode(w).mode[0])
                new_y_test.append([ label ])
            y_test = np.array( new_y_test )

            new_y_val = []
            for w in y_val:
                label = int(stats.mstats.mode(w).mode[0])
                new_y_val.append([ label ])
            y_val = np.array( new_y_val )

        return x_train, y_train, x_test, y_test, x_val, y_val



class PPGSubject(Subject):
    def __init__(self, path, id, sensor, device, experiment_type, window_duration=30):
        Subject.__init__(self, path=path, id=id, device=device, sensor=sensor, experiment_type=experiment_type)
        self._filtered = self.filtered(self._data['ppg'], [0.1, 9], 2)
        self._peaks = self.peaks(self._filtered)
        self._peak_label = [self._data['y'][p] for p in self._peaks]
        self._bpm = self.bpm()
        self._ibi = self.ibi()
        self._mean_bpm = self.mean_bpm()
        self._median_bpm = self.median_bpm()
        self.window_duration = window_duration
        # self._dx1 = self._compute_dx(self._filtered)
        # self._ms_points = self.peaks(self._dx1)
        # self._dx2 = self._compute_dx(self._dx1)
        # self._dx3 = self._compute_dx(self._dx2)

    def _values(self):
        result_x = []
        result_y = []
        X, y = self.statistical()
        for i in range(0, len(X), self.window_duration):
            wx = X[i:i+30]
            wy = y[i:i+30]
            missing_on_window = self.window_duration - len(wx)
            if missing_on_window > 0:
                wx = np.concatenate([wx, np.zeros((missing_on_window, np.shape(wx)[1]))])
                wy = np.concatenate([wy,np.zeros(missing_on_window)])

            result_x.append(wx)
            result_y.append(wy)
            self._computed_x = result_x
        self._computed_y = result_y
        return result_x, result_y
    
    def x(self):
        if self.experiment_type == ExperimentType.END_TO_END:
            return super().x()
        else:
            if len(self._computed_x) > 0:
                return self._computed_x
            _x, _ = self._values()
            return _x

    def y(self):
        if self.experiment_type == ExperimentType.END_TO_END:
            return super().y()
        else:
            if len(self._computed_y) > 0:
                return self._computed_y
            _, _y = self._values()
            return _y


    def statistical(self):
        x = np.array([*zip(self._bpm, self._ibi, self._mean_bpm, self._median_bpm, *self._stats())])
        x = MinMaxScaler().fit_transform(x)
        return x, self._peak_label

    def _stats(self):
        std = []
        mean = []
        median = []
        max = []
        min = []
        windows = self.peak_data_window()
        for w in windows:
            std.append(np.std(w))
            mean.append(np.mean(w))
            median.append(np.median(w))
            min.append(np.min(w))
            max.append(np.max(w))
        return std, mean, median, max, min


    def peak_data_window(self):
        result = []
        start = 0
        for i in range(0, len(self._peaks)):
            end = self._peaks[i]+1 # first data point for the next window
            window = self._data['ppg'][start:end]
            result.append(window)
            start = end
        return result

    def show_filtered(self, window, wn, n):
        self._filtered = self.filtered(self._data['ppg'], n, wn)
        start, finish = self.window(window)
        data = self._filtered[start:finish]
        peaks = self.peaks(data)
        peak_val = [data[x] for x in peaks]
        plt.figure(figsize=(25, 4))
        plt.plot(peaks, peak_val, 'ro')
        plt.plot(data)
        plt.show()

    def window(self, w):
        ws = self._x.sampling * 30
        w_start = w * ws
        return w_start, w_start + ws

    def peaks(self, data) -> list[int]:
        peaks_x, _ = sig.find_peaks(data, distance=self.min_peak_distance(), height=0.0)
        return peaks_x

    def min_peak_distance(self):
        return 0.6 * self._x.sampling

    def filtered(self, data, wn, n):
        return filter_signal(data, self._x.sampling, wn, n)

    def bpm(self):
        bpm = []
        for i in range(0, len(self._peaks) - 1):
            bpm.append(self._x.sampling / abs(self._peaks[i] - self._peaks[i+1]) *  60)
        # get last bpm with leftover data
        last_bpm = (self._x.sampling / len(self._filtered) - self._peaks[len(self._peaks) - 1]) * 60
        bpm.append(np.mean([ bpm[len(bpm) -1], last_bpm, bpm[len(bpm) -1] ]))
        return bpm
    
    def ibi(self):
        ibi = []
        for i in range(0, len(self._peaks)):
            if i == 0:
                ibi.append(self._peaks[i] / self._x.sampling)
            else:
                ibi.append((self._peaks[i] - self._peaks[i-1]) / self._x.sampling)
        return ibi

    def mean_bpm(self):
        mean = []
        for i in range(0, len(self._bpm)):
            if i == 0:
                mean.append(np.mean([self._bpm[0], self._bpm[0], self._bpm[1]]))
            elif i == len(self._bpm) -1:
                mean.append(np.mean([self._bpm[len(self._bpm) -2], self._bpm[len(self._bpm) -1], self._bpm[len(self._bpm) -1]]))
            else:
                mean.append(np.mean([self._bpm[i - 1], self._bpm[i], self._bpm[i+1]]))
        return mean

    def median_bpm(self):
        median = []
        for i in range(0, len(self._bpm)):
            if i == 0:
                median.append(np.median([self._bpm[0], self._bpm[0], self._bpm[1]]))
            elif i == len(self._bpm) -1:
                median.append(np.median([self._bpm[len(self._bpm) -2], self._bpm[len(self._bpm) -1], self._bpm[len(self._bpm) -1]]))
            else:
                median.append(np.median([self._bpm[i - 1], self._bpm[i], self._bpm[i+1]]))
        return median




    # def show_dx1(self,w):
    #     start = self._ms_points[w]
    #     finish = self._ms_points[w + 1]
    #     data = [self._filtered[p] for p in range(start, finish)]
    #     plt.figure(figsize=(25, 4))
    #     plt.plot(0, self._filtered[start], 'ro')
    #     plt.plot(data)
    #     plt.show()
    #
    # def show_dx(self, w, dxn=1):
    #     data = self._dx1
    #     if dxn == 2:
    #         data = self._dx2
    #     if dxn == 3:
    #         data = self._dx3
    #     start, finish = self.window(w)
    #     w_peaks = []
    #     for p in self.peaks(data):
    #         if start <= p <= finish:
    #             w_peaks.append(p)
    #     peaks_v = [data[p] for p in w_peaks]
    #     peak_x = [dxp - start for dxp in w_peaks]
    #
    #     plt.figure(figsize=(25, 4))
    #     plt.plot(peak_x, peaks_v, 'ro')
    #     plt.plot(data[start:finish])
    #     plt.show()
    #
    # def _compute_dx(self, x) -> list[float]:
    #     dx = []
    #     for i in range(1, len(x) -1):
    #         dx.append((x[i-1] - x[i+1]) * self._x.sampling)
    #     dx.insert(0, dx[0])
    #     dx.append(dx[len(dx) - 1])
    #     return dx
    #
    # def _point_dx(self, p1: float, p2: float):
    #     return p1 - p2


class PPGWindow():
    def __init__(self, signal: Signal, x: int, y):
        self.signal: Signal = signal
        self.x = signal.data
        self.index = x
        self.y = y
        self.window_overlap_ratio = 0.5
        self._dx1 = []
        self._dx2 = []
        self._dx3 = []
        self._x_peaks = []
        self._bpm = pd.DataFrame()

    def min_peak_distance(self):
        """
            The distance between peaks translates into the max frequency accepted for heart beats.
            The current value used allows for (roughly) up to 2.2 beats per second, or 133bpm (which is a lot).

            A higher number is used in here due to the high noise on the Samsung device collected data, therefore
            making it possible to better estimate where the peaks are.
        """
        return 0.45 * self.signal.sampling

    def peaks(self) -> list[int]:
        if len(self._x_peaks) == 0: 
            self._x_peaks, _ = sig.find_peaks(self.x, distance=self.min_peak_distance(), height=0.0)
        return self._x_peaks

    def dx_peaks(self):
        dx, _, _ = self.dx()
        peaks, _ = sig.find_peaks(dx, distance=self.min_peak_distance(), height=0.0)
        return peaks

    def dx_valleys(self):
        dx, _, _ = self.dx()
        peaks, _ = sig.find_peaks([x*-1 for x in dx], distance=self.min_peak_distance(), height=0.0)
        return peaks

    def dicrotic_notch(self):
        self.peaks()

    def dx(self):
        """
            Derivation for the data signal, should only be used with filtered data.
        """
        if len(self._dx1) == 0 and len(self._dx2) == 0 and len(self._dx3) == 0:
            dx1 = self._compute_dx(self.x, scaleTime=True)
            dx2 = self._compute_dx(dx1)
            dx3 = self._compute_dx(dx2)
            self._dx1 = dx1
            self._dx2 = dx2
            self._dx3 = dx3

        return self._dx1, self._dx2, self._dx3

    def _compute_dx(self, x, scaleTime=False) -> list[float]:
        dx = []
        for i in range(1, len(x) -1):
            dx.append(self._point_dx(x[i-1], x[i+1], scaleTime=scaleTime))
        dx.insert(0, dx[0])
        dx.append(dx[len(dx) - 1])
        return dx

    def show(self):
        plt.figure(figsize=(25, 4))
        plt.plot(self.x)
        plt.show()

    def _point_dx(self, p1: float, p2: float, scaleTime=False):
        if scaleTime:
            return (p1 - p2) * self.signal.sampling
        return p1 - p2

    def bpm(self):
        if self._bpm.empty:
            bpm = []
            for i in range(0, len(self.peaks()) - 1):
                bpm.append(self.signal.sampling / abs(self.peaks()[i] - self.peaks()[i+1]) *  60)
            self._bpm['bpm'] = bpm
        return self._bpm

    def zeig_dich(self):
        dx1, dx2, dx3 = self.dx()
        peaks = self.peaks()
        _, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(25,16))

        ax1.plot(self.x)
        peak_val = [self.x[x] for x in peaks]
        ax1.plot(peaks, peak_val, 'ro')
        #ax1.vlines(peaks, ymin=min(self.x), ymax=max(self.x), color="C9")
        ax1.legend(["PPG", "Peaks"], loc="upper right")

        ax2.plot(dx1, color="C1")
        max_slope = self.dx_peaks()
        min_slope = self.dx_valleys()
        ax2.plot(max_slope, [dx1[x] for x in max_slope], 'ro')
        ax2.plot(min_slope, [dx1[x] for x in min_slope], 'bo')
        ax2.legend(["PPG'", "ms", "mins"], loc="upper right")

        ax3.plot(dx2, color="C3")
        ax3.legend(["PPG''"], loc="upper right")

        ax4.plot(dx3, color="C5",label="dx3")
        ax4.legend(["PPG'''"], loc="upper right")
        #plt.title(label="{0} Window".format(self.index))
        plt.show()


