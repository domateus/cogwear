import os
from src.classifiers.shallow import ShallowClassifier
from xgboost import XGBClassifier

class Xgb(ShallowClassifier):
    def build_model(self, hyperparameters):
        print(f'hyperparameters: {hyperparameters}')
        return XGBClassifier(**hyperparameters)

    def feature_importance(self, path):
        model = self.load_model(path)
        return model.get_score(importance_type='gain')
