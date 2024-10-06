from keras import models, layers
from src.classifiers.classifier import Classifier

class Lstm(Classifier):
    def build_model(self, input_shape):

        if self.hyperparameters:
            lstm_units = self.hyperparameters.lstm_units
            filters_multipliers = self.hyperparameters.filters_multipliers
            kernel_size_multipliers = self.hyperparameters.kernel_size_multiplier
        else:
            lstm_units = 4
            filters_multipliers = 1
            kernel_size_multipliers = 0.5

        input_layers = []
        output_layers = []

        shape = input_shape if len(input_shape) == 2 else input_shape[1:]
        cols = 1 if len(input_shape) == 2 else input_shape[0]

        for _ in range(0, cols):
            input_layer = layers.Input(shape)
            input_layers.append(input_layer)

            assert filters_multipliers is not None
            assert kernel_size_multipliers is not None

            filters_1 = int(filters_multipliers * 4)
            filters_2 = int(filters_multipliers * 8)
            kernel_size_1 = int(kernel_size_multipliers * 4)
            kernel_size_2 = int(kernel_size_multipliers * 8)

            # length of convolution window (kernel size) cannot be larger than number of steps
            conv_layer = layers.Conv1D(filters=filters_1, kernel_size=kernel_size_1)(input_layer)
            conv_layer = layers.MaxPooling1D(pool_size=2)(conv_layer)
            conv_layer = layers.Conv1D(filters=filters_2, kernel_size=kernel_size_2, padding='same')(conv_layer)
            conv_layer = layers.MaxPooling1D(pool_size=2)(conv_layer)

            output_layers.append(conv_layer)

        concat = layers.concatenate(output_layers, axis=-1) if len(output_layers) > 1 else output_layers[0]
        flatten_layer = layers.TimeDistributed(layers.Flatten())(concat)
        dense_layer = layers.TimeDistributed(layers.Dense(lstm_units))(flatten_layer)
        lstm_layer = layers.LSTM(lstm_units)(dense_layer)
        output_layer = layers.Dense(1, activation='sigmoid')(lstm_layer)

        model = models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(loss='binary_crossentropy', optimizer=self.get_optimizer(), metrics=['precision', 'recall', 'F1Score'])

        return model

