import tensorflow as tf

class NoiseLayer(tf.keras.layers.Layer):
    def __init__(self, noise_std):
        super(NoiseLayer, self).__init__()
        self.noise_std = noise_std

    def call(self, inputs, training=None, mask=None):
        if training:
            return inputs + tf.random.normal(shape=inputs.shape, stddev=self.noise_std, dtype=tf.float64)
        else:
            return inputs
