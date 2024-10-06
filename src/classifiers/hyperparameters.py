
class Hyperparameters():
    def __init__(self, lr_power, decay, reduce_lr_factor, batch_size, filters_multipliers=None, filters=None, kernel_size_multiplier=None, kernel_sizes=None, dense_outputs=None, depth=None, lstm_units=None, baseline_weight=1):
        self.filters_multipliers = filters_multipliers
        self.filters = filters
        self.kernel_size_multiplier = kernel_size_multiplier
        self.kernel_sizes = kernel_sizes
        self.dense_outputs = dense_outputs
        self.depth = depth
        self.lstm_units = lstm_units
        self.batch_size = batch_size
        self.epochs = 300
        self.lr = 1 / 10 ** lr_power
        self.decay = decay
        self.reduce_lr_factor = reduce_lr_factor
        self.reduce_lr_patience = 30
        self.class_weights = {0: 1*baseline_weight, 1: 1}


    def dict(self):
        return self.__dict__
