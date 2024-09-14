import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
from sklearn.preprocessing import MinMaxScaler

def filter_signal(signal, sampling):
    sos = scipy.signal.iirfilter(4, Wn=[0.7, 3.0], fs=sampling, btype="bandpass",
                             ftype="butter", output="sos")
    result = scipy.signal.sosfilt(sos, signal)
    # result = nk.signal_detrend(signal)
    # result = nk.signal_filter(result, lowcut=0.7, highcut=3.0)
    result = pd.Series(result).iloc[::1]
    result = np.array(result).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(result)
    result = scaler.transform(result)
    return result.reshape(1, -1)[0]
