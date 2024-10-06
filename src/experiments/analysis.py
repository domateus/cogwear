import pandas as pd
import os
from src.classifiers.tuning.hptuning import Tuner
from src.experiments.experiment import  Experiment
from src.experiments.consts import ExperimentType
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Analysis(Experiment):
    def __init__(self, signal: str, type: ExperimentType, path: str, device:str):
        self.aha = 42
        self.type = type
        self.device = device
        self.signal = signal
        self.test_metrics_path =  os.path.join(path, "results", self.type.name, f"{self.device}_{self.signal}", "test")
        self.fcn = Experiment(signal=signal, classifier="fcn",type=type, device=device, path=path)
        self.cnn = Experiment( signal=signal, classifier="cnn",type=type, device=device, path=path)
        self.lstm = Experiment(signal=signal,classifier="lstm",type=type,device=device,path=path)
        self.resnet = Experiment(signal=signal,classifier="resnet",type=type,device=device,path=path)
        self.t_fcn = Tuner(self.fcn)
        self.t_cnn = Tuner(self.cnn)
        self.t_lstm = Tuner(self.lstm)
        self.t_resnet = Tuner(self.resnet)

    def list_metrics(self, skip=[]):
        files = {"fcn": {}, "cnn": {}, "lstm": {}, "resnet":{}}
        for file in os.listdir(self.test_metrics_path):
          if not os.path.isdir(os.path.join(self.test_metrics_path, file)):
            if file.startswith("fcn") and "fcn" not in skip:
              files["fcn"][file[3:5]] = os.path.join(self.test_metrics_path, file)
            if file.startswith("cnn") and "cnn" not in skip:
              files['cnn'][file[3:5]] = os.path.join(self.test_metrics_path, file)
            if file.startswith("lstm") and "lstm" not in skip:
              files['lstm'][file[4:6]] = os.path.join(self.test_metrics_path, file)
            if file.startswith("resnet") and "resnet" not in skip:
              files['resnet'][file[6:8]] = os.path.join(self.test_metrics_path, file)

        for classifier in files:
          df_classifier = pd.DataFrame()
          for subject in files[classifier]:
            metrics_path = files[classifier][subject]
            df_sub = pd.read_csv(metrics_path)
            df_classifier = pd.concat([df_classifier,df_sub.assign(subject=subject, classifier=classifier)])
          files[classifier]["df"] = df_classifier

        return files

    def head(self, num=3, skip=[]):
        files = self.list_metrics(skip)
        for c in files:
          if isinstance(files[c]["df"], pd.DataFrame) and 'f1' in files[c]["df"]:
            print(files[c]["df"].sort_values(by="f1", ascending=False).head(num))

    def compare(self, skip=[]):
        files = self.list_metrics(skip)
        for c in files:
          if isinstance(files[c]["df"], pd.DataFrame) and 'f1' in files[c]["df"]:
            subject = files[c]["df"].sort_values(by="f1", ascending=False).head(1)["subject"].values[0]
            img_name = f'{subject}predictions.png'
            print(os.path.join(self.test_metrics_path, c, img_name))
            img = mpimg.imread(os.path.join(self.test_metrics_path, c, img_name))
            plt.imshow(img)
            plt.show()
