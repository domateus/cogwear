import os
import time
from abc import ABC, abstractmethod
import numpy as np
from keras import optimizers, callbacks, backend, models

from src.classifiers.hyperparameters import Hyperparameters
from src.logger import log_predicions, logger, save_logs


class Classifier(ABC):
    def __init__(self, output_directory, input_shape, hyperparameters: Hyperparameters, verbose=False, model_init: models.Model | None = None, fold=-1, name=""):
        self.output_directory = output_directory
        self.verbose = verbose
        self.fold = fold
        self.name = name
        self.hyperparameters = hyperparameters
        self.callbacks = []
        self.best_model_path = os.path.join(output_directory, '-1_best_model.weights.h5')
        self.model = model_init if model_init else self.build_model(input_shape)
        assert self.model
        if verbose:
            self.model.summary()
        self.create_callbacks()

    @abstractmethod
    def build_model(self, input_shape) -> models.Model:
        pass

    def create_callbacks(self):
        backup = callbacks.BackupAndRestore(backup_dir="/tmp/backup")
        self.callbacks.append(backup)
        model_checkpoint = callbacks.ModelCheckpoint(filepath=os.path.join(self.output_directory, f"{self.fold}_best_model.weights.h5"), monitor='val_loss', save_best_only=True, save_weights_only=True)
        self.callbacks.append(model_checkpoint)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=self.hyperparameters.reduce_lr_factor, patience=self.hyperparameters.reduce_lr_patience)
        self.callbacks.append(reduce_lr)
        early_stopping = callbacks.EarlyStopping(patience=30)
        self.callbacks.append(early_stopping)

    def get_optimizer(self):
        return optimizers.AdamW(learning_rate=self.hyperparameters.lr, weight_decay=self.hyperparameters.decay)

    def fit(self, x_train, y_train, x_val, y_val, y_true, batch_size=16, nb_epochs=500, x_test=None, shuffle=True):
        mini_batch_size = int(min(x_train[0].shape[0] / 10, batch_size))
        logger.info(f"Fitting model: {self.name}, shape: {x_train[0].shape}")
        start_time = time.time()
        hist = self.model.fit(x_train, y_train, class_weight=self.hyperparameters.class_weights, batch_size=mini_batch_size, epochs=nb_epochs, verbose='2' if self.verbose else '1', validation_data=(x_val, y_val), callbacks=self.callbacks, shuffle=shuffle)
        duration = time.time() - start_time
        y_pred_probabilities = self.model.predict(x_test)
        y_pred = np.round(y_pred_probabilities)
        metrics, loss = save_logs(self.output_directory, hist, y_pred, y_true, duration, self.fold)
        backend.clear_session()
        return metrics, loss

    def predict(self, x_test, y_true, model_path, round):
        print(f'predicting')
        self.model.load_weights(model_path)
        print(f'loaded weights')
        start_time = time.time()
        y_pred_probabilities = self.model.predict(x_test)
        duration = time.time() - start_time
        y_pred = np.round(y_pred_probabilities)
        metrics = log_predicions(self.output_directory, y_pred, y_true, duration, self.fold, round)
        backend.clear_session()
        return metrics
