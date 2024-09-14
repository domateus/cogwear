from keras import layers, models
from src.classifiers.classifier import Classifier


class Cnn(Classifier):
    def build_model(self, input_shape):
        input_layers = []

        filters_multipliers = 1
        kernel_size_multipliers = 1

        if self.hyperparameters:
            filters_multipliers = self.hyperparameters.filters_multipliers
            kernel_size_multipliers = self.hyperparameters.kernel_size_multiplier

        assert kernel_size_multipliers is not None
        assert filters_multipliers is not None
        padding = 'valid'
        input_layer = layers.Input(input_shape)
        input_layers.append(input_layer)

        kernel_size = int(kernel_size_multipliers * 7)

        conv1 = layers.Conv1D(filters=int(filters_multipliers * 6), kernel_size=kernel_size,
                                    padding=padding, activation='relu')(input_layer)
        conv1 = layers.AveragePooling1D(pool_size=3)(conv1)

        conv2 = layers.Conv1D(filters=int(filters_multipliers * 12), kernel_size=kernel_size,
                                    padding=padding, activation='relu')(conv1)
        conv2 = layers.AveragePooling1D(pool_size=3)(conv2)

        flatten_layer = layers.Flatten()(conv2)

        output_layer = layers.Dense(1, activation='sigmoid')(flatten_layer)

        model = models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(loss='binary_crossentropy', optimizer=self.get_optimizer(), metrics=['precision', 'recall', 'F1Score'])

        return model
