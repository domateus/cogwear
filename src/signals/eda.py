from src.signals.subject import Subject
from src.classifiers.svm import Svm
from random import randrange
from src.data.utils import Split
from src.classifiers.xgb import Xgb
from src.classifiers.knn import Knn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import scipy.stats as stats
import neurokit2 as nk
from src.experiments.experiment import Experiment
from src.experiments.consts import  ExperimentType

class EDAExperiment(Experiment):
    def __init__(self, path: str, type: ExperimentType, classifier: str, window_duration=30):
        Experiment.__init__(self, signal="eda", classifier=classifier,type=type, device="empatica", path=path, window_duration=window_duration, subject=EDASubject)
        self.device = "empatica"
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

class EDASubject(Subject):
    def __init__(self, path, id, sensor, device, experiment_type, window_duration=30):
        Subject.__init__(self, path=path, id=id, device='empatica', sensor='eda', experiment_type=experiment_type)
        self.window_duration = window_duration

    def values(self):
        if ExperimentType.END_TO_END == self.experiment_type:
            print(f'right values')
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
        signal, _ = nk.eda_process(self._data['eda'], sampling_rate=4)
        signal = signal.fillna(0)
        ws = self._x.sampling * self.window_duration
        data = []
        ys = []
        for x in range(0, len(self._data['eda']), ws):
            w = signal[x:x+ws].values
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
