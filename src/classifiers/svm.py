from src.classifiers.shallow import ShallowClassifier
from sklearn.svm import SVC

class Svm(ShallowClassifier):
    def build_model(self,hyperparameters):
        print(f'hyperparameters: {hyperparameters}')
        return SVC(C=hyperparameters['C'], class_weight=hyperparameters['class_weight'], degree=hyperparameters['degree'], gamma=hyperparameters['gamma'], kernel=hyperparameters['kernel'])
