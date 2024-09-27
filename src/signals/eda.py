from src.experiments.experiment import Experiment, ExperimentType

class EDAExperiment(Experiment):
    def __init__(self, path: str, type: ExperimentType, classifier: str, window_duration=30, pilot=False):
        Experiment.__init__(self, signal="eda", classifier=classifier,type=type, device="empatica", path=path, window_duration=window_duration, pilot=pilot)
        self.device = "empatica"
        self.path = path
