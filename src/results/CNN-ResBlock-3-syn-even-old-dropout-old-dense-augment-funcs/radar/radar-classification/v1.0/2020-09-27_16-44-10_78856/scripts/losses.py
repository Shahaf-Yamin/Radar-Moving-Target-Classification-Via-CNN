import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, binary_crossentropy, SparseCategoricalCrossentropy, CategoricalCrossentropy
from backend.loss_metric_utils import nan_mask, identity


def step_function_new(y_i, y_j):
    diff = y_j - y_i
    return tf.math.multiply(tf.constant(-1.),
                            tf.math.log(tf.math.divide(tf.constant(1.), tf.constant(1.) + tf.math.exp(diff))))

def auc_max(y_true, y_pred):
    """
    y_true = ground truth values with shape = `[batch_size, d0, .. dN]`
    y_pred = predicted values with shape = `[batch_size, d0, .. dN]`
    """

    negative_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    positive_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    i_pos = 0
    i_neg = 0

    for i in tf.range(tf.constant(64)):
        label = y_true[i]
        if label == tf.constant(1,dtype=tf.int8):
            positive_array = positive_array.write(i_pos, y_pred[i])
            i_pos = i_pos + 1
        else:
            negative_array = negative_array.write(i_neg, y_pred[i])
            i_neg = i_neg + 1

    L = tf.constant(0., shape=(1,))

    for i in tf.range(tf.constant(64)):
        label = y_true[i]
        L_t = tf.constant(0.,shape = (1,))
        if label == tf.constant(1,dtype=tf.int8):
            for j in tf.range(negative_array.size()):
                L_t += (1. / tf.cast(negative_array.size(),dtype=tf.float32)) * step_function_new(y_pred[i], negative_array.read(j))
        else:
            for j in tf.range(positive_array.size()):
                L_t += (1. / tf.cast(positive_array.size(),dtype=tf.float32) ) * step_function_new(positive_array.read(j), y_pred[i])
        L += L_t

    return L


class ClassificationLoss(tf.keras.losses.Loss):

    def __init__(self, name='classification_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = BinaryCrossentropy()

        self.weight_fn = nan_mask

        self.target_fn = identity

        self.pred_fn = identity

    def call(self, targets, prediction, sample_weight=None):
        targets = tf.cast(tf.reshape(targets, [-1, 1]), tf.float32)
        prediction = tf.cast(tf.reshape(prediction, [-1, prediction.shape[-1]]), tf.float32)

        tar = self.target_fn(targets)
        pred = self.pred_fn(prediction)
        weights = self.weight_fn(targets)
        loss = self.loss_fn(tar, pred, sample_weight=weights)

        return loss

class AUCLoss(tf.keras.losses.Loss):
    def __init__(self, regularization_factor=0.1, name="custom_mse"):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        # loss = tf.math.reduce_mean(tf.square(tf.cast(y_true, tf.float32) - y_pred))
        loss = auc_max(y_true, y_pred)
        # reg = tf.math.reduce_mean(tf.square(tf.constant(0.5,dtype=tf.float32) - y_pred))
        return loss
        # return mse + reg * self.regularization_factor
