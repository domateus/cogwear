import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
from sklearn.preprocessing import MinMaxScaler

def filter_signal(signal, sampling, wn, n):
    result = scipy.stats.mstats.winsorize(signal, limits=[0.01, 0.01])
    sos = scipy.signal.iirfilter(n, Wn=wn, fs=sampling, btype="bandpass",
                             ftype="butter", output="sos")
    result = scipy.signal.sosfilt(sos, result)
    result = pd.Series(result).iloc[::1]
    result = np.array(result).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(result)
    result = scaler.transform(result)
    return result.reshape(1, -1)[0]
