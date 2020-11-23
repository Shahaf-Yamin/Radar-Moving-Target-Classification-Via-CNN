import numpy as np
import tensorflow as tf
import logging

logger = logging.getLogger("logger")

# def scheduler(epoch, lr):
#   if (epoch + 1) % 10 == 0:
#     return lr / 2
#   else:
#     return lr

# def scheduler(epoch, lr):
#   if epoch < 5:
#     return 1e-4
#   elif epoch < 10:
#     return 1e-5
#   elif epoch < 17:
#     return 1e-6
#   elif epoch < 22:
#     return 1e-7
#   else:
#     return 1e-8

class LRScheduler(object):
    def __init__(self, lr_init, lr_factor, lr_patience):
        self.lr_init = lr_init
        self.factor = lr_factor
        self.lr_patience = lr_patience

    def scheduler(self,epoch, lr):
        if epoch < self.lr_patience:
            return self.lr_init
        elif epoch < 2 * self.lr_patience:
            return self.factor * self.lr_init
        elif epoch < 3 * self.lr_patience:
            return (self.factor ** 2) * self.lr_init
        else:
            return (self.factor ** 3) * self.lr_init


def scheduler(epoch, lr):
    if epoch < 10:
        return 1e-4
    elif epoch < 20:
        return 1e-5
    elif epoch < 30:
        return 1e-6
    elif epoch < 40:
        return 1e-7
    else:
        return 1e-8

class HalfCycleLrScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def get_config(self):
        pass

    def __init__(self, lr, max_steps, **kwargs):
        super().__init__(**kwargs)
        self.max_steps = max_steps
        self.lr = lr

    def __call__(self, step):
        if step % 10:
            self.lr /= 2
        return self.lr
#       return self.lr * (1 + tf.math.sin(tf.constant(np.pi, dtype=tf.float64) * step / (self.max_steps * 0.75)))

