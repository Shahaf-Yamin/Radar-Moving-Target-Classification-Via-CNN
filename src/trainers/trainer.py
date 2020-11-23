import tensorflow as tf
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
import logging
from metrics.metrics import ClassificationMetrics
from optimizers.lr_schedulers import scheduler, LRScheduler, AUCLRScheduler
from losses.losses import *
# from data.data_loader import generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
import os
import pickle
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.keras import balanced_batch_generator

logger = logging.getLogger("logger")


def build_trainer(model, data, config, exp_name_time,sweep_string=None):
    if config.trainer_name == "classification":
        # trainer = ClassificationTrainer(model, data, config)
        trainer = ClassificationTrainerKeras(model, data, config, exp_name_time,sweep_string)
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
    def __init__(self, model, data, config, exp_name_time,sweep_string=None):
        self.model_train = model['train']
        # self.model_eval = model['eval']
        self.data = data
        self.config = config
        self.exp_name_time = exp_name_time
        self.sweep_string = sweep_string
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

    def get_optimizer(self):
        def get_Adam():
            use_amsgrad = True if self.config.optimizer == "AdamAmsGrad" else False
            if self.config.Adam_eps == "default":
                Adam_eps = 1e-7
            elif type(self.config.Adam_eps) == float:
                Adam_eps = self.config.Adam_eps
            else:
                raise Exception('Unsupported Adam_eps!! --> is not default/float')
            return Adam(learning_rate=self.config.learning_rate, epsilon=Adam_eps, amsgrad=use_amsgrad)
        def get_Nadam():
            if self.config.Adam_eps == "default":
                Adam_eps = 1e-7
            elif type(self.config.Adam_eps) == float:
                Adam_eps = self.config.Adam_eps
            else:
                raise Exception('Unsupported Adam_eps!! --> is not default/float')
            return Nadam(learning_rate=self.config.learning_rate, epsilon=Adam_eps)


        optimizer_name = self.config.optimizer
        if optimizer_name == "Adam" or optimizer_name == "AdamAmsGrad":
            return get_Adam()
        elif optimizer_name == "Nadam":
            return get_Nadam()
        else:
            raise Exception('Unsupported optimizer!!')



    def train(self):
        self.model_train.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=self.metrics)
        callback_list = []
        
        if self.config.lr_scheduler == "lr_scheduler":
            lr_schedule_obj = LRScheduler(self.config.learning_rate, self.config.lr_scheduler_factor, self.config.lr_patience)
            callback_list.append(tf.keras.callbacks.LearningRateScheduler(lr_schedule_obj.scheduler, verbose=1))
        elif self.config.lr_scheduler == "plateau":
            callback_list.append(ReduceLROnPlateau(monitor='val_accuracy', factor=self.config.lr_scheduler_factor, verbose=1,
                                                   min_lr=(self.config.lr_scheduler_factor ** 4) * self.config.learning_rate,
                                                   patience=self.config.lr_patience))
        elif self.config.lr_scheduler == "auc_scheduler":
            callback_list.append(AUCLRScheduler(lr_init=self.config.learning_rate, lr_factor=self.config.lr_scheduler_factor, lr_patience=self.config.lr_patience, auc_thr=self.config.auc_thr))
        # model checkpoint to save best model
        checkpoint_dir = './Checkpoints/checkpoint_{}'.format(self.exp_name_time)
        checkpoint_best_filepath = '{}/model_checkpoint_best'.format(checkpoint_dir)
        checkpoint_epoch_filepath = '{}/model_checkpoint_epoch'.format(checkpoint_dir)
        if self.config.model_checkpoint:
            assert self.config.model_checkpoint_mode == 'min' or self.config.model_checkpoint_mode == 'max'
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            # save best model
            save_model_best_callback = ModelCheckpoint(filepath=checkpoint_best_filepath, save_weights_only=False,
                                                  monitor=self.config.model_checkpoint_metric, mode=self.config.model_checkpoint_mode,
                                                  save_best_only=True, verbose=1)
            callback_list.append(save_model_best_callback)
        if self.config.save_model_periodically:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            # save model periodically
            save_model_epoch_callback = ModelCheckpoint(filepath=checkpoint_epoch_filepath, save_weights_only=False,
                                                    save_best_only=False, verbose=1, period=self.config.save_model_period)
            callback_list.append(save_model_epoch_callback)

        # early stopping
        if self.config.use_early_stop:
            early_stopping = EarlyStopping(monitor=self.config.early_stop_metric, restore_best_weights=True,
                                           patience=self.config.early_stop_patience, verbose=1)
            callback_list.append(early_stopping)

        # CSV Logger for per-epoch logging
        if self.config.save_fit_history:
            exp_graph_dir = '../graphs/{}'.format(self.exp_name_time)
            if self.sweep_string is None:
                csv_logger_path = '{}/{}_fit_log.csv'.format(exp_graph_dir, self.exp_name_time)
            else:
                csv_logger_path = '{}/{}_{}_fit_log.csv'.format(exp_graph_dir, self.exp_name_time,self.sweep_string)
            csv_logger = CSVLogger(csv_logger_path)
            callback_list.append(csv_logger)

        # if self.config.with_rect_augmentation is True:
        #     history = self.model_train.fit(x=generator(data=self.data['train'], config=self.config),
        #                                    steps_per_epoch=self.config.steps_per_epoch,
        #                                    epochs=self.config.num_epochs,
        #                                    validation_data=self.data['train_eval'],
        #                                    callbacks=callback_list, verbose=self.config.fit_verbose)
        # else:
        steps_per_epoch = self.config.steps_per_epoch if self.config.steps_per_epoch_overwrite else None
        callback_list = None if callback_list == [] else callback_list

        history = self.model_train.fit(self.data['train'], epochs=self.config.num_epochs,
                                       validation_data=self.data['train_eval'],
                                       callbacks=callback_list,
                                       verbose=self.config.fit_verbose, steps_per_epoch= steps_per_epoch)
        # if self.config.model_checkpoint:
        #     self.model_train = tf.keras.models.load_model(checkpoint_filepath)

        # save fit history
        if self.config.save_fit_history:
            exp_graph_dir = '../graphs/{}'.format(self.exp_name_time)
            if self.sweep_string is None:
                history_fit_path = '{}/{}_fit_history_final.pkl'.format(exp_graph_dir, self.exp_name_time)
            else:
                history_fit_path = '{}/{}_{}_fit_history_final.pkl'.format(exp_graph_dir, self.exp_name_time,self.sweep_string)
            if os.path.exists(exp_graph_dir):
                with open(history_fit_path, 'wb') as file:
                    pickle.dump(history.history, file, pickle.HIGHEST_PROTOCOL)
                    print(10 * '#')
                    print('Fit history dumped to {}'.format(history_fit_path))
                    # reference code for loading the file later into a dict
                    # with open(history_fit_path, 'rb') as file:
                    #     history_load = pickle.load(file)
            else:
                print(10 * '#')
                print('Could not dump fit history to {}'.format(history_fit_path))

        return history

    def evaluate(self):
        eval_res = self.model_train.evaluate(self.data['train_eval'])
        return eval_res
