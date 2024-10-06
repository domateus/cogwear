from keras import layers, models
from src.classifiers.classifier import Classifier


class Fcn(Classifier):
    def build_model(self, input_shape):
        input_layers = []
        output_layers = []

        shape = input_shape if len(input_shape) == 2 else input_shape[1:]
        cols = 1 if len(input_shape) == 2 else input_shape[0]

        for i in range(0, cols):
            input_layer = layers.Input(shape)
            input_layers.append(input_layer)

            conv1 = layers.Conv1D(filters=int(128 * self.hyperparameters.filters_multipliers),kernel_size=int(8 * self.hyperparameters.kernel_size_multiplier), padding='same')(input_layer)
            conv1 = layers.BatchNormalization()(conv1)
            conv1 = layers.Activation(activation='relu')(conv1)

            conv2 = layers.Conv1D(filters=int(256 * self.hyperparameters.filters_multipliers),kernel_size=int(5 * self.hyperparameters.kernel_size_multiplier), padding='same')(conv1)
            conv2 = layers.BatchNormalization()(conv2)
            conv2 = layers.Activation('relu')(conv2)

            conv3 = layers.Conv1D(int(128 * self.hyperparameters.filters_multipliers),kernel_size=int(3 * self.hyperparameters.kernel_size_multiplier), padding='same')(conv2)
            conv3 = layers.BatchNormalization()(conv3)
            conv3 = layers.Activation('relu')(conv3)

            gap_layer = layers.GlobalAveragePooling1D()(conv3)
            output_layers.append(gap_layer)

        flat = layers.concatenate(output_layers, axis=-1) if len(output_layers) > 1 else output_layers[0]

        output_layer = layers.Dense(1, activation='sigmoid')(flat)

        model = models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(loss='binary_crossentropy', optimizer=self.get_optimizer(), metrics=['precision', 'recall', 'F1Score'])

        return model
