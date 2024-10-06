from src.classifiers.shallow import ShallowClassifier
from sklearn.neighbors import KNeighborsClassifier

class Knn(ShallowClassifier):
    def build_model(self, hyperparameters):
        print(f'hyperparameters: {hyperparameters}')
        return KNeighborsClassifier(**hyperparameters)
