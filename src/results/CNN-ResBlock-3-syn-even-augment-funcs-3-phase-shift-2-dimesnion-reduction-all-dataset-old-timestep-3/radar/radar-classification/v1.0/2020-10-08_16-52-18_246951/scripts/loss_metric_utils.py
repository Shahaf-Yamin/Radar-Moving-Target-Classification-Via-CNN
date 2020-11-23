import tensorflow as tf
import logging

logger = logging.getLogger("logger")


def nan_mask(x):
    x = tf.cast(x, tf.float32)
    return tf.logical_not(tf.math.is_nan(x))

def nan_mask_5_7(x):
    x = tf.cast(x, tf.float32)
    return tf.logical_and(tf.logical_not(tf.math.is_nan(x)),
                          tf.logical_or(tf.equal(x,5),tf.equal(x,7)))

def identity(x):
    return x

def round(x):
    return tf.round(x)

def argmax(x):
    return tf.argmax(x, axis=-1)
