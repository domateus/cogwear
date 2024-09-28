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
from src.classifiers.classifier import Classifier
from src.data.utils import LOSOCV_SUBJECT_IDS, SUBJECTS_IDS, TEST_SUBJECT_IDS, losocv_splits
from src.signals.subject import Subject
from src.signals.utils import filter_signal
from src.signals.signal import Signal
from src.experiments.experiment import Experiment, ExperimentType

class PPG():
    def __init__(self, path):
        print(path)

class PPGExperiment(Experiment):
    def __init__(self, path: str, type: ExperimentType, classifier: str, device="samsung", window_duration=30, pilot=False):
        Experiment.__init__(self, signal="ppg", classifier=classifier,type=type, device= device, path=path, window_duration=window_duration, pilot=pilot)
        self.device = device
        self.path = path


class PPGSubject(Subject):
    def __init__(self, path, subject_id, sensor, device):
        Subject.__init__(self, path=path, id=subject_id, device=device, sensor=sensor)
        self._filtered = self.filtered(self._data['ppg'])

    def show_filtered(self, window):
        return self.show(window, self.filtered())

    def filtered(self, data):
        return filter_signal(data, self._x.sampling)

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


