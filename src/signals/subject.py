import scipy.stats as stats
import numpy as np
from abc import ABC
from math import floor
import os
import pandas as pd
from src.signals.signal import Signal

class Subject(ABC):
    def __init__(self, path: str, id: str, device: str, sensor: str, window_duration=30):
        self.path = path
        self.device = device
        self.sampling = Subject.get_device_sampling(device, sensor)
        self.sensor = sensor
        self.id = id
        self._data = self.load()
        self._x = self._get_signal()
        self._y = self._data["y"]
        self.x, self.y = self.values(seconds=window_duration)

    def _get_signal(self):
        return Signal(self.sensor, self.sensor, self.sampling, [self._data[self.sensor]])

    def _get_window(self, index=0, seconds=30, overlap_ratio=0.5):
        window_size = seconds * self._x.sampling
        initial_point = floor(index * window_size * (1 - overlap_ratio))
        y = int(stats.mstats.mode(self._y[initial_point:initial_point+window_size]).mode[0])
        x = [it for it in self._x.data[0][initial_point:initial_point+window_size]]
        for _ in range(0, window_size - len(x)):
            x.append(0)
        return x, y

    def _expand_dims_axis(self) -> int:
        return 1

    def values(self, seconds=30, overlap_ratio=0):
        window_size = seconds * self._x.sampling
        step = window_size * (1 - overlap_ratio)
        windows_count = int(len(self._data) // step)
        X = []
        Y = []
        for index in range(0, windows_count):
            x, y = self._get_window(index=index, seconds=seconds, overlap_ratio=overlap_ratio)
            X.append(np.expand_dims(x, self._expand_dims_axis()))
            Y.append([y])
        return X, Y

    @staticmethod
    def get_device_sampling(device="samsung", sensor="ppg"):
        if device == "muse":
            return 256
        elif device == "samsung":
            return 25

        return 64 if sensor == "ppg" else 4

    def load(self):
        return pd.read_csv(os.path.join(self.path, self.id, self.device + f"_{self.sensor}.csv"))
