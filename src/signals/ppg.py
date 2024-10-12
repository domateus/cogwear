from heartpy.analysis import clean_rr_intervals, calc_rr, calc_fd_measures
from heartpy.peakdetection import check_peaks
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import scipy.signal as sig
from sklearn.preprocessing import MinMaxScaler
from src.classifiers.svm import Svm
from src.classifiers.xgb import Xgb
from src.classifiers.knn import Knn
from src.signals.subject import Subject
from src.signals.utils import filter_signal
from src.experiments.consts import  ExperimentType
from src.experiments.experiment import Experiment
from random import randrange

class PPGExperiment(Experiment):
    def __init__(self, path: str, type: ExperimentType, classifier: str, device="samsung", window_duration=30):
        Experiment.__init__(self, signal="ppg", classifier=classifier,type=type, device= device, path=path, window_duration=window_duration, subject=PPGSubject)
        self.device = device
        self.path = path

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

class PPGSubject(Subject):
    def __init__(self, path, id, sensor, device, experiment_type, window_duration=30):
        Subject.__init__(self, path=path, id=id, device=device, sensor=sensor, experiment_type=experiment_type)
        self._filtered = self.filtered(self._data['ppg'], [0.1, 9], 2)
        self._peaks = self.peaks(self._filtered)
        self._peak_label = [self._data['y'][p] for p in self._peaks]
        self.window_duration = window_duration

    def hp_values(self, peaks):
        wd = calc_rr(peaks, sample_rate=self._x.sampling)
        ybeat = [self._filtered[p] for p in peaks]
        wd = check_peaks(wd['RR_list'], peaks, ybeat, working_data=wd)
        wd, m = calc_fd_measures(working_data=wd)
        wd = clean_rr_intervals(wd)
        return wd, m

    def all_windows(self):
        ws = self._x.sampling * self.window_duration
        data = []
        ys = []
        for x in range(0, len(self._filtered), ws):
            w = [v for v in self._filtered[x:x+ws]]
            y = [v for v in self._data['y'][x:x+ws]]
            for _ in range(0, ws - len(w)):
                w.append(0)
                y.append(0)
            label = int(stats.mstats.mode(y).mode[0])
            ys.append(label)
            data.append(w)
        return data, ys

    def values(self):
        if ExperimentType.END_TO_END ==self.experiment_type:
            return super().values()
        ws, ys = self.all_windows()
        result = []
        for w in ws:
            peaks = self.peaks(w)
            bpm = self.bpm(peaks)
            ibi = self.ibi(peaks)
            wd, _ = self.hp_values(peaks)
            row = []
            for v in self.sts(wd['RR_list']):
                row.append(v)
            for v in self.sts(wd['RR_diff']):
                row.append(v)
            for v in self.sts(wd['RR_sqdiff']):
                row.append(v)
            for v in self.sts(wd['RR_list_cor']):
                row.append(v)
            for v in self.sts(wd['frq']):
                row.append(v)
            for v in self.sts(wd['psd']):
                row.append(v)
            for v in self.sts(bpm):
                row.append(v)
            for v in self.sts(ibi):
                row.append(v)
            for v in self.sts(w):
                row.append(v)
            row.append(len(wd['removed_beats']))
            row.append(len(peaks))
            result.append(row)
        result = MinMaxScaler().fit_transform(result)
        return result, ys

    def show_filtered(self, window, wn, n):
        self._filtered = self.filtered(self._data['ppg'], n, wn)
        start, finish = self.window_bounds(window)
        data = self._filtered[start:finish]
        peaks = self.peaks(data)
        peak_val = [data[x] for x in peaks]
        plt.figure(figsize=(25, 4))
        plt.plot(peaks, peak_val, 'ro')
        plt.plot(data)
        plt.show()

    def window_bounds(self, w):
        ws = self._x.sampling * self.window_duration
        w_start = w * ws
        return w_start, w_start + ws

    def peaks(self, data) -> list[int]:
        peaks_x, _ = sig.find_peaks(data, distance=self.min_peak_distance(), height=0.0)
        #delete first peak if within first 150ms (signal might start mid-beat after peak)
        return [p for p in peaks_x if p > self._x.sampling * 150/1000]

    def min_peak_distance(self):
        return 0.6 * self._x.sampling

    def filtered(self, data, wn, n):
        return filter_signal(data, self._x.sampling, wn, n)

    def bpm(self, peaks):
        bpm = []
        for i in range(0, len(peaks) - 1):
            bpm.append(self._x.sampling / abs(peaks[i] - peaks[i+1]) *  60)
        return bpm
    
    def ibi(self, peaks):
        ibi = [peaks[0] / self._x.sampling]
        for i in range(1, len(peaks)):
            ibi.append((peaks[i] - peaks[i-1]) / self._x.sampling)
        return ibi
