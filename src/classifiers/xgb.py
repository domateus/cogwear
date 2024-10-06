from src.classifiers.shallow import ShallowClassifier
from xgboost import XGBClassifier

class Xgb(ShallowClassifier):
    def build_model(self, hyperparameters):
        print(f'hyperparameters: {hyperparameters}')
        return XGBClassifier(**hyperparameters)
