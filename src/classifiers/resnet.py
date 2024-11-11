from keras import layers, models
from src.classifiers.classifier import Classifier


class OneDResNet(Classifier):
    def build_model(self, input_shape):
        n_feature_maps = 64

        depth = self.hyperparameters.depth if self.hyperparameters and self.hyperparameters.depth else 3
        filter = self.hyperparameters.filters if self.hyperparameters and self.hyperparameters.filters else n_feature_maps

        input_layers = []
        output_layers = []

        shape = input_shape if len(input_shape) == 2 else input_shape[1:]
        cols = 1 if len(input_shape) == 2 else input_shape[0]
        assert self.hyperparameters.kernel_size_multiplier is not None
        for _ in range(0, cols):
            current_layer = layers.Input(shape)
            input_layers.append(current_layer)

            for i_depth in range(depth - 1):
                mult = 2 if i_depth > 0 else 1
                current_layer = self.build_bloc(int(mult * filter), current_layer)
            # BLOCK LAST
            conv_x = layers.Conv1D(filters=int(filter * 2),
                                         kernel_size=int(self.hyperparameters.kernel_size_multiplier * 8), padding='same')(
                current_layer)
            conv_x = layers.BatchNormalization()(conv_x)
            conv_x = layers.Activation('relu')(conv_x)

            conv_y = layers.Conv1D(filters=int(filter * 2),
                                         kernel_size=int(self.hyperparameters.kernel_size_multiplier * 5), padding='same')(
                conv_x)
            conv_y = layers.BatchNormalization()(conv_y)
            conv_y = layers.Activation('relu')(conv_y)

            conv_z = layers.Conv1D(filters=int(filter * 2), kernel_size=int(self.hyperparameters.kernel_size_multiplier * 3), padding='same')(
                conv_y)
            conv_z = layers.BatchNormalization()(conv_z)

            shortcut_y = current_layer
            if depth == 2:
                shortcut_y = layers.Conv1D(filters=filter * 2, kernel_size=1, padding='same')(shortcut_y)
            shortcut_y = layers.BatchNormalization()(shortcut_y)

            output_block_3 = layers.add([shortcut_y, conv_z])
            output_block_3 = layers.Activation('relu')(output_block_3)

            # FINAL 
            gap_layer = layers.GlobalAveragePooling1D()(output_block_3)
            output_layers.append(gap_layer)

        flat = layers.concatenate(output_layers, axis=-1) if len(output_layers) > 1 else output_layers[0]
        output_layer = layers.Dense(1, activation='sigmoid')(flat)

        model = models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(loss='binary_crossentropy', optimizer=self.get_optimizer(), metrics=['precision', 'recall', 'F1Score'])

        return model

    def build_bloc(self, n_feature_maps, input_layer):
        conv_x = layers.Conv1D(filters=n_feature_maps, kernel_size=int(self.hyperparameters.kernel_size_multiplier * 8),
                                     padding='same')(input_layer)
        conv_x = layers.BatchNormalization()(conv_x)
        conv_x = layers.Activation('relu')(conv_x)

        conv_y = layers.Conv1D(filters=n_feature_maps, kernel_size=int(self.hyperparameters.kernel_size_multiplier * 5),
                                     padding='same')(conv_x)
        conv_y = layers.BatchNormalization()(conv_y)
        conv_y = layers.Activation('relu')(conv_y)

        conv_z = layers.Conv1D(filters=n_feature_maps, kernel_size=int(self.hyperparameters.kernel_size_multiplier * 3),
                                     padding='same')(conv_y)
        conv_z = layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = layers.BatchNormalization()(shortcut_y)

        output_block_1 = layers.add([shortcut_y, conv_z])
        return layers.Activation('relu')(output_block_1)
