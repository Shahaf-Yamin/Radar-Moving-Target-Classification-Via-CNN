import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import logging
from metrics.metrics import ClassificationMetrics
from optimizers.lr_schedulers import scheduler
from losses.losses import *
from data.data_loader import generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence

# from imblearn.over_sampling import RandomOverSampler
# from imblearn.keras import balanced_batch_generator

logger = logging.getLogger("logger")


def build_trainer(model, data, config):
    if config.trainer_name == "classification":
        # trainer = ClassificationTrainer(model, data, config)
        trainer = ClassificationTrainerKeras(model, data, config)
    else:
        raise ValueError("'{}' is an invalid model name")

    return trainer


# class ClassificationTrainer:
#     def __init__(self, model, data, config):
#         self.model_train = model['train']
#         self.model_test = model['eval']
#         self.data = data
#         self.config = config
#
#         self.loss_fn = ClassificationLoss(config)
#         self.optimizer = Adam(learning_rate=self.config.learning_rate)
#
#         self.metric_train = ClassificationMetrics(config.train_writer, name='train')
#         self.metric_train_eval = ClassificationMetrics(config.train_writer, name='train_eval')
#         self.metric_test = ClassificationMetrics(config.test_writer, name='test')
#
#         self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
#
#     @tf.function
#     def compute_grads(self, samples, targets):
#         with tf.GradientTape() as tape:
#             predictions = self.model_train(samples, training=True)
#
#             ''' generate the targets and apply the corresponding loss function '''
#
#             loss = self.loss_fn(targets, predictions)
#
#         gradients = tape.gradient(loss, self.model_train.trainable_weights)
#         gradients, grad_norm = tf.clip_by_global_norm(gradients, self.config.clip_grad_norm)
#         with self.config.train_writer.as_default():
#             tf.summary.scalar("grad_norm", grad_norm, self.global_step)
#             self.global_step.assign_add(1)
#
#         return gradients, predictions
#
#     @tf.function
#     def apply_grads(self, gradients):
#         self.optimizer.apply_gradients(zip(gradients, self.model_train.trainable_weights))
#
#     def sync_eval_model(self):
#         model_weights = self.model_train.get_weights()
#         ma_weights = self.model_test.get_weights()
#         alpha = self.config.moving_average_coefficient
#         self.model_test.set_weights([ma * alpha + w * (1 - alpha) for ma, w in zip(ma_weights, model_weights)])
#
#     @tf.function
#     def train_step(self, samples, targets):
#         gradients, predictions = self.compute_grads(samples, targets)
#         self.apply_grads(gradients)
#         return predictions
#
#     @tf.function
#     def eval_step(self, samples):
#
#         predictions = self.model_test(samples, training=False)
#
#         return predictions
#
#     def train_epoch(self, epoch):
#         self.metric_train.reset_states()
#         self.model_train.reset_states()
#
#         for samples, targets in self.data['train']:
#             predictions = self.train_step(samples, targets)
#             self.metric_train.update_state(targets, predictions)
#             self.sync_eval_model()
#
#         self.metric_train.print(epoch)
#         self.metric_train.log_metrics(epoch)
#
#     def evaluate_train(self, epoch):
#         self.metric_train_eval.reset_states()
#         self.model_test.reset_states()
#
#         for samples, targets in self.data['train_eval']:
#             predictions = self.eval_step(samples)
#             self.metric_train_eval.update_state(targets, predictions)
#
#         self.metric_train_eval.print(epoch)
#         self.metric_train_eval.log_metrics(epoch)
#
#     def evaluate_test(self, epoch):
#         self.metric_test.reset_states()
#         self.model_test.reset_states()
#
#         for samples, targets in self.data['test']:
#             predictions = self.eval_step(samples)
#             self.metric_test.update_state(targets, predictions)
#
#         self.metric_test.print(epoch)
#         self.metric_test.log_metrics(epoch)
#
#     def train(self):
#         for epoch in range(self.config.num_epochs):
#             self.train_epoch(epoch)
#
#             if epoch % self.config.eval_freq == 0:
#                 self.evaluate_train(epoch)
#                 self.evaluate_test(epoch)


# class BalancedDataGenerator(Sequence):
#     """ImageDataGenerator + RandomOversampling"""
#     def __init__(self, data, data_gen, batch_size=32):
#         self.datagen = data_gen
#         self.batch_size = min(batch_size, x.shape[0])
#         data_gen.fit()
#
#         self.gen, self.steps_per_epoch = balanced_batch_generator(x.reshape(x.shape[0], -1), y, sampler=RandomOverSampler(),
#                                                                   batch_size=self.batch_size, keep_sparse=True)
#         self._shape = (self.steps_per_epoch * batch_size, *x.shape[1:])
#
#     def __len__(self):
#         return self.steps_per_epoch
#
#     def __getitem__(self, idx):
#         x_batch, y_batch = self.gen.__next__()
#         x_batch = x_batch.reshape(-1, *self._shape[1:])
#         return self.datagen.flow(x_batch, y_batch, batch_size=self.batch_size).next()


class ClassificationTrainerKeras(object):
    def __init__(self, model, data, config):
        self.model_train = model['train']
        # self.model_eval = model['eval']
        self.data = data
        self.config = config

        # self.loss_fn = BinaryCrossentropy()
        # self.loss_fn = auc_max

        if config.learn_background:
            self.loss_fn = 'categorical_crossentropy'
        elif config.loss == "AUC":
            self.loss_fn = AUCLoss()
        elif config.loss == "BinaryCrossentropy":
            self.loss_fn = BinaryCrossentropy()
        else:
            self.loss_fn = None

        self.optimizer = Adam(learning_rate=self.config.learning_rate)
        self.metrics = ['accuracy', AUC()]

        # self.metric_train = ClassificationMetrics(config.train_writer, name='train')
        # self.metric_train_eval = ClassificationMetrics(config.train_writer, name='train_eval')
        # self.metric_test = ClassificationMetrics(config.test_writer, name='test')

        # self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

    def train(self):
        self.model_train.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=self.metrics)

        if self.config.callback != "None":
            callback_list = []
            if self.config.callback == "lr_scheduler":
                callback_list.append(tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1))
            elif self.config.callback == "plateau":
                callback_list.append(ReduceLROnPlateau(monitor='val_accuracy', factor=0.7, verbose=1,
                                                       min_lr=0.05 * self.config.learning_rate,
                                                       patience=self.config.lr_patience))
            # model checkpoint to save best wieghts
            checkpoint_filepath = './model_checkpoint'
            if self.config.model_checkpoint:
                save_model_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True,
                                                      monitor='val_accuracy', mode='max', save_best_only=True)
                callback_list.append(save_model_callback)
            # early stopping
            if self.config.use_early_stop:
                early_stopping = EarlyStopping(monitor=self.config.early_stop_metric, restore_best_weights=True,
                                               patience=self.config.early_stop_patience)
                callback_list.append(early_stopping)

            if self.config.with_rect_augmentation is True:
                history = self.model_train.fit(x=generator(data=self.data['train'], config=self.config),
                                               steps_per_epoch=self.config.steps_per_epoch,
                                               epochs=self.config.num_epochs,
                                               validation_data=self.data['train_eval'],
                                               callbacks=callback_list, verbose=self.config.fit_verbose)
            else:
                history = self.model_train.fit(self.data['train'], epochs=self.config.num_epochs,
                                               validation_data=self.data['train_eval'],
                                               callbacks=callback_list, verbose=self.config.fit_verbose)
            if self.config.model_checkpoint:
                self.model_train.load_weights(checkpoint_filepath)
        else:
            history = self.model_train.fit(self.data['train'], epochs=self.config.num_epochs,
                                           validation_data=self.data['train_eval'])

        return history

    def evaluate(self):
        eval_res = self.model_train.evaluate(self.data['train_eval'])
        return eval_res
