import os
import shutil
import numpy as np
from src.classifiers.classifier import Classifier
from src.classifiers.fcn import Fcn
from src.classifiers.cnn import Cnn
from src.classifiers.resnet import OneDResNet
from src.classifiers.lstm import Lstm
from src.signals.subject import Subject
import itertools as it
import random

SUBJECTS_IDS = list(it.chain(range(11, 19), range(20, 25)))
LOSOCV_SUBJECT_IDS = [SUBJECTS_IDS[i] for i in range(0, 10)]
TEST_SUBJECT_IDS = [SUBJECTS_IDS[i] for i in range(10, 13)]

class Split():
    def __init__(self, id, train, test, val):
        self.id = id
        self.train = train
        self.test = test
        self.val = val

    def x_train(self):
        return  np.concatenate([s.x() for s in self.train])

    def y_train(self):
        return np.concatenate([s.y() for s in self.train])

    def x_test(self):
        return np.concatenate([s.x() for s in self.test])

    def y_test(self):
        return np.concatenate([s.y() for s in self.test])

    def x_val(self):
        return np.concatenate([s.x() for s in self.val])

    def y_val(self):
        return np.concatenate([s.y() for s in self.val])


    class Pre():
        def __init__(self, id, train, test, val):
            self.id = id
            self.train = train
            self.test = test
            self.val = val

        def into(self, subjects: list[Subject]):
            train = [s for s in subjects if s.id in self.train]
            test = [s for s in subjects if s.id in self.test]
            val = [s for s in subjects if s.id in self.val]
            return Split(self.id, train, test, val)

def losocv_splits() -> list[Split.Pre]:
    result = []
    subjects = LOSOCV_SUBJECT_IDS 
    for subject in subjects:
        test_set = [f"{subject}"]
        rest = [f"{x}" for x in subjects if not x == subject]
        val_set = random.sample(rest, 1)
        train_set = [x for x in rest if x not in val_set]
        result.append(Split.Pre(id=subject, train=train_set, test=test_set, val=val_set))
    return result


def create_classifier(classifier_name, input_shape, output_directory, hyperparameters, fold) -> Classifier:
    if classifier_name == 'fcn':
        return Fcn(output_directory, input_shape, hyperparameters=hyperparameters, fold=fold, name=classifier_name)
    if classifier_name == 'cnn':
        return Cnn(output_directory, input_shape, hyperparameters=hyperparameters, fold=fold, name=classifier_name)
    if classifier_name == 'lstm':
        return Lstm(output_directory, input_shape, hyperparameters=hyperparameters, fold=fold, name=classifier_name)
    if classifier_name == 'resnet':
        return OneDResNet(output_directory, input_shape, hyperparameters=hyperparameters, fold=fold, name=classifier_name)
    return Fcn(output_directory, input_shape, hyperparameters=hyperparameters, fold=fold, name=classifier_name)


def wipe_results():
    shutil.rmtree(os.path.join(os.getcwd(), 'results'))
