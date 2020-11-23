import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from backend.loss_metric_utils import nan_mask, identity


class ClassificationLoss(tf.keras.losses.Loss):

    def __init__(self, config, name='classification_loss', **kwargs):
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
