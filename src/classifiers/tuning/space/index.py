from abc import ABC, abstractmethod

class Space(ABC):
    def __init__(self, classifier):
        self.c = classifier

    @abstractmethod
    def dict(self, hyperparameters):
        pass

    @abstractmethod
    def get_hyperparameters(self, x):
        pass

    @abstractmethod
    def get_search_space(self):
        pass

    @abstractmethod
    def best_trial_hyperparameter(self, trials):
        pass

