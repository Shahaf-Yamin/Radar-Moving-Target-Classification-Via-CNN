import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import ReduceLROnPlateau
import logging
from losses.losses import ClassificationLoss
from metrics.metrics import ClassificationMetrics
from optimizers.lr_schedulers import scheduler

logger = logging.getLogger("logger")


def build_trainer(model, data, config):
    if config.trainer_name == "classification":
        # trainer = ClassificationTrainer(model, data, config)
        trainer = ClassificationTrainerKeras(model, data, config)
    else:
        raise ValueError("'{}' is an invalid model name")

    return trainer


class ClassificationTrainer:
    def __init__(self, model, data, config):
        self.model_train = model['train']
        self.model_test = model['eval']
        self.data = data
        self.config = config

        self.loss_fn = ClassificationLoss(config)
        self.optimizer = Adam(learning_rate=self.config.learning_rate)

        self.metric_train = ClassificationMetrics(config.train_writer, name='train')
        self.metric_train_eval = ClassificationMetrics(config.train_writer, name='train_eval')
        self.metric_test = ClassificationMetrics(config.test_writer, name='test')

        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

    @tf.function
    def compute_grads(self, samples, targets):
        with tf.GradientTape() as tape:
            predictions = self.model_train(samples, training=True)

            ''' generate the targets and apply the corresponding loss function '''

            loss = self.loss_fn(targets, predictions)

        gradients = tape.gradient(loss, self.model_train.trainable_weights)
        gradients, grad_norm = tf.clip_by_global_norm(gradients, self.config.clip_grad_norm)
        with self.config.train_writer.as_default():
            tf.summary.scalar("grad_norm", grad_norm, self.global_step)
            self.global_step.assign_add(1)

        return gradients, predictions

    @tf.function
    def apply_grads(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.model_train.trainable_weights))

    def sync_eval_model(self):
        model_weights = self.model_train.get_weights()
        ma_weights = self.model_test.get_weights()
        alpha = self.config.moving_average_coefficient
        self.model_test.set_weights([ma * alpha + w * (1 - alpha) for ma, w in zip(ma_weights, model_weights)])

    @tf.function
    def train_step(self, samples, targets):
        gradients, predictions = self.compute_grads(samples, targets)
        self.apply_grads(gradients)
        return predictions

    @tf.function
    def eval_step(self, samples):

        predictions = self.model_test(samples, training=False)

        return predictions

    def train_epoch(self, epoch):
        self.metric_train.reset_states()
        self.model_train.reset_states()

        for samples, targets in self.data['train']:
            predictions = self.train_step(samples, targets)
            self.metric_train.update_state(targets, predictions)
            self.sync_eval_model()

        self.metric_train.print(epoch)
        self.metric_train.log_metrics(epoch)

    def evaluate_train(self, epoch):
        self.metric_train_eval.reset_states()
        self.model_test.reset_states()

        for samples, targets in self.data['train_eval']:
            predictions = self.eval_step(samples)
            self.metric_train_eval.update_state(targets, predictions)

        self.metric_train_eval.print(epoch)
        self.metric_train_eval.log_metrics(epoch)

    def evaluate_test(self, epoch):
        self.metric_test.reset_states()
        self.model_test.reset_states()

        for samples, targets in self.data['test']:
            predictions = self.eval_step(samples)
            self.metric_test.update_state(targets, predictions)

        self.metric_test.print(epoch)
        self.metric_test.log_metrics(epoch)

    def train(self):
        for epoch in range(self.config.num_epochs):
            self.train_epoch(epoch)

            if epoch % self.config.eval_freq == 0:
                self.evaluate_train(epoch)
                self.evaluate_test(epoch)


class ClassificationTrainerKeras(object):
    def __init__(self, model, data, config):
        self.model_train = model['train']
        self.model_eval = model['eval']
        self.data = data
        self.config = config

        self.loss_fn = BinaryCrossentropy()
        self.optimizer = Adam(learning_rate=self.config.learning_rate)
        self.metrics = ['accuracy', AUC()]

        # self.metric_train = ClassificationMetrics(config.train_writer, name='train')
        # self.metric_train_eval = ClassificationMetrics(config.train_writer, name='train_eval')
        # self.metric_test = ClassificationMetrics(config.test_writer, name='test')

        # self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

    def train(self):
        self.model_train.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=self.metrics)

        if self.config.callback != "None":
            if self.config.callback == "lr_scheduler":
                callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
            elif self.config.callback == "plateau":
                callback = ReduceLROnPlateau(monitor='val_accuracy', factor=0.7, verbose=1,
                                             min_lr=0.01 * self.config.learning_rate, patience=7)
            history = self.model_train.fit(self.data['train'], epochs=self.config.num_epochs,
                                           validation_data=self.data['train_eval'],
                                           callbacks=[callback])
        else:
            history = self.model_train.fit(self.data['train'], epochs=self.config.num_epochs,
                                           validation_data=self.data['train_eval'])



        return history

    def evaluate(self):
        self.model_train.evaluate(self.data['train_eval'])
