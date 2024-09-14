from src.experiments.experiment import Experiment, ExperimentType

class EDAExperiment(Experiment):
    def __init__(self, path: str, type: ExperimentType, classifier: str):
        Experiment.__init__(self, signal="eda", classifier=classifier,type=type, device="empatica", path=path)
        self.device = "empatica"
        self.path = path
