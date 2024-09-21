# pyright can be annoying
from src.signals.subject import Subject, Signal
from math import floor
import scipy.stats as stats

class EEGSubject(Subject):
    def __init__(self, path, id, device, sensor):
        self._eeg_cols = ['Alpha_TP9','Alpha_AF7','Alpha_AF8','Alpha_TP10','Beta_TP9','Beta_AF7','Beta_AF8','Beta_TP10','Gamma_TP9','Gamma_AF7','Gamma_AF8','Gamma_TP10','Theta_TP9','Theta_AF7','Theta_AF8','Theta_TP10','Delta_TP9','Delta_AF7','Delta_AF8','Delta_TP10']

        Subject.__init__(self, path=path, id=id, device="muse", sensor="eeg")

    def _get_signal(self, discard_time):
        data = [self._data[col][discard_time:-discard_time] for col in self._eeg_cols]
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
    
