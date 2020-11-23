import numpy as np
import tensorflow as tf
import logging

logger = logging.getLogger("logger")

def scheduler(epoch, lr):
  if epoch % 10 == 0:
    return lr / 2
  else:
    return lr

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

