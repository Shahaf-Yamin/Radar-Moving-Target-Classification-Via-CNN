import tensorflow as tf
import logging
from tensorflow.keras.metrics import AUC
from backend.loss_metric_utils import nan_mask, identity, round
logger = logging.getLogger("logger")

class ClassificationMetrics(tf.keras.metrics.Metric):

    def __init__(self, writer, name='', **kwargs):
        super().__init__(name=name, **kwargs)
        self.writer = writer
        self.metric_pool = [tf.keras.metrics.BinaryCrossentropy(name='BCE'.format(name)),
                            Accuracy(name='accuracy'.format(name)),
                            AUC(name='AUC'.format(name))]

        self.weight_fn = [nan_mask,
                          nan_mask,
                          nan_mask]

        self.target_fn = [identity,
                          identity,
                          identity]

        self.pred_fn = [identity,
                          round,
                          identity]

    def update_state(self, targets, prediction, sample_weight=None):
        for metric, weight_fn, pred_fn, target_fn  in zip(self.metric_pool, self.weight_fn, self.pred_fn, self.target_fn):
            metric.update_state(target_fn(targets), pred_fn(prediction), sample_weight=weight_fn(targets))

    def result(self):
        return [metric.result() for metric in self.metric_pool]

    def reset_states(self):
        for metric in self.metric_pool:
            metric.reset_states()
        return

    def print(self, epoch):
        msg = ["{:20s} {:05d}\t".format("{} epoch:".format(self.name), epoch)]
        for metric in self.metric_pool:
            msg.append("{:8s} {:3.5f}\t".format("{}".format(metric.name), float(metric.result())))
        logger.info("\t".join(msg))

    def log_metrics(self, epoch):
        with self.writer.as_default():
            for metric in self.metric_pool:
                tf.summary.scalar(metric.name, metric.result(), epoch)


class MCC(tf.keras.metrics.Metric):
    """ Matthews correlation coefficient:
        Computed using the following formula
        MCC = (TP x TN - FP x FN) / ( (TP+FP)(TP+FN)(TN+FP)(TN+FN) )
        where TP - true positives,
              TN - true negatives,
              FP - false positives,
              FN - false negatives
    """
    def __init__(self, name='profit', **kwargs):
        super(MCC, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

        self.pos_symbol = tf.constant(1, dtype=tf.int64)
        self.neg_symbol = tf.constant(-1, dtype=tf.int64)

    def update_state(self, targets, prediction, sample_weight=None):
        targets = tf.cast(tf.squeeze(targets), tf.int64)
        prediction = tf.cast(tf.squeeze(prediction), tf.int64)


        values = tf.cast(tf.equal(prediction,targets), dtype=tf.int64)
        not_values = tf.cast(tf.not_equal(prediction,targets), dtype=tf.int64)


        tp = tf.where(tf.equal(prediction, tf.ones_like(prediction) * self.pos_symbol),
                      values, tf.zeros_like(values))
        tn = tf.where(tf.equal(prediction, tf.ones_like(prediction) * self.neg_symbol),
                      values, tf.zeros_like(values))
        fp = tf.where(tf.equal(prediction, tf.ones_like(prediction) * self.pos_symbol),
                      not_values, tf.zeros_like(values))
        fn = tf.where(tf.equal(prediction, tf.ones_like(prediction) * self.neg_symbol),
                      not_values, tf.zeros_like(values))

        if sample_weight is not None:
            sample_weight = tf.cast(tf.squeeze(sample_weight), tf.int64)
            tp = tf.multiply(tp, sample_weight)
            tn = tf.multiply(tn, sample_weight)
            fp = tf.multiply(fp, sample_weight)
            fn = tf.multiply(fn, sample_weight)

        tp = tf.cast(tf.reduce_sum(tp), dtype=tf.float32)
        tn = tf.cast(tf.reduce_sum(tn), dtype=tf.float32)
        fp = tf.cast(tf.reduce_sum(fp), dtype=tf.float32)
        fn = tf.cast(tf.reduce_sum(fn), dtype=tf.float32)

        self.tp.assign(self.tp + tp)
        self.tn.assign(self.tn + tn)
        self.fp.assign(self.fp + fp)
        self.fn.assign(self.fn + fn)

    def result(self):
        denum = tf.sqrt((self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn))
        denum = tf.where(denum == 0, 1., denum)
        num = self.tp * self.tn - self.fp * self.fn
        return num / denum

class Accuracy(tf.keras.metrics.Metric):

    def __init__(self, name='accuracy', **kwargs):
        super(Accuracy, self).__init__(name=name, **kwargs)
        self.true = self.add_weight(name='true', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, targets, prediction, sample_weight=None):
        targets = tf.cast(tf.squeeze(targets), tf.int64)
        prediction = tf.cast(tf.squeeze(prediction), tf.int64)


        values = tf.cast(tf.equal(prediction, targets), dtype=tf.float32)

        if sample_weight is not None:
            sample_weight = tf.cast(tf.squeeze(sample_weight), tf.float32)
            self.true.assign(self.true + tf.reduce_sum(tf.multiply(values, sample_weight)))
            self.total.assign(self.total + tf.reduce_sum(sample_weight))
        else:
            self.true.assign(self.true + tf.reduce_sum(values))
            self.total.assign(self.total + values.shape[0])

    def result(self):
        return self.true / self.total
