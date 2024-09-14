from keras import layers, models
from src.classifiers.classifier import Classifier


class Fcn(Classifier):
    def build_model(self, input_shape):

        input_layer = layers.Input(input_shape)

        conv1 = layers.Conv1D(filters=int(128 * self.hyperparameters.filters_multipliers),kernel_size=int(8 * self.hyperparameters.kernel_size_multiplier), padding='same')(input_layer)
        conv1 = layers.BatchNormalization()(conv1)
        conv1 = layers.Activation(activation='relu')(conv1)

        conv2 = layers.Conv1D(filters=int(256),kernel_size=int(5), padding='same')(conv1)
        conv2 = layers.BatchNormalization()(conv2)
        conv2 = layers.Activation('relu')(conv2)

        conv3 = layers.Conv1D(int(128),kernel_size=int(3), padding='same')(conv2)
        conv3 = layers.BatchNormalization()(conv3)
        conv3 = layers.Activation('relu')(conv3)

        gap_layer = layers.GlobalAveragePooling1D()(conv3)
        output_layer = layers.Dense(1, activation='sigmoid')(gap_layer)

        model = models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='binary_crossentropy', optimizer=self.get_optimizer(), metrics=['precision', 'recall', 'F1Score'])

        return model
