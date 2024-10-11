from src.experiments.experiment import Experiment
from src.classifiers.tuning.space import dl, shallow
import os
import pickle
import tensorflow as tf
from src.experiments.experiment import Experiment
from hyperopt import fmin, tpe,  Trials, STATUS_OK
from keras.api.backend import clear_session
from src.classifiers.tuning.utils import is_shallow


class Tuner():
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.logger = experiment.logger
        self.trial_path = experiment.trials_path
        os.makedirs(self.trial_path, exist_ok=True)
        self.space = shallow.Shallow(experiment.classifier) if is_shallow(experiment.classifier) else dl.Deep(experiment.classifier)

    def tune(self, evals, max_evals):
        trials = self.load_trials()
        start = len(trials)
        finish = min(start + evals, max_evals)
        self.logger.info(f"trials length: {len(trials)}")
        space = self.space.get_search_space()
        for i in range(start, finish):
            fmin(fn=lambda x: self._objective(x, i),
                 space=space,
                 algo=tpe.suggest,
                 max_evals=i + 1,
                 trials=trials)

            # Moved into the loop due to preemptible gpu
            self.save_trials(trials)
        return trials

    def _objective(self, x, iteration):
        self.logger.info(f"Running objective for {self.experiment.classifier} at {iteration} iteration")
        hp = self.space.get_hyperparameters(x)
        with tf.device('/device:GPU:0'):
            clear_session()
            _, loss = self.experiment.run_once(hp, percentage_data=.2)
            clear_session()

            

        return {"status": STATUS_OK,
                "x": self.space.dict(hp),
                "loss": loss}
            
    def best_hyperparameters(self):
        return self.space.best_trial_hyperparameter(self.load_trials())

    def load_trials(self):
        trials_filename = os.path.join(self.trial_path, self.experiment.classifier + ".pkl")

        try:
            if os.path.exists(trials_filename):
                with open(trials_filename, "rb") as f:
                    self.logger.info(f"loaded existing trials")
                    return pickle.load(f)
        except EOFError:
            pass

        trials = Trials()
        self.save_trials(trials)
        return trials

    def save_trials(self, trials):
        with open(os.path.join(self.trial_path, self.experiment.classifier + ".pkl"), "wb") as f:
            pickle.dump(trials, f)
