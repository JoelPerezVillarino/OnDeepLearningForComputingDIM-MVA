import tensorflow as tf


class AutomaticLrScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, monitor, factor, patience, threshold, min_lr):
        super(AutomaticLrScheduler, self).__init__()
        self.lr = initial_lr
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.wait = 0
        self.best_loss = float("inf")
    
    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor)

        # if current_loss < self.best_loss:
        if current_loss < self.best_loss * self.threshold:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait+=1
            if self.wait >= self.patience:
                self.lr = max(self.lr*self.factor, self.min_lr)
                print(f"\nLearning rate reduced to: {self.lr:.7e}\n")
                self.model.optimizer.lr = self.lr
                self.wait = 0
