#!/usr/bin/env python3

from models.models import build_model
from utils.utils import preprocess_meta_data
from main_train import test_model
import tensorflow as tf
import os
import datetime
import matplotlib
from datetime import datetime
from data.data_parser import DataSetParser
from data.data_loader import expand_test_by_sampling_rect
import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC

matplotlib.use('Agg')

tf.keras.backend.set_floatx('float32')


def adjust_input_size(config):
    if config.with_iq_matrices is True or config.with_magnitude_phase is True:
        config.__setattr__("model_input_dim", [126, 32, 2])
    if config.with_rect_augmentation:
        config.__setattr__("model_input_dim", [126, config.rect_augment_num_of_timesteps, 1])
    if config.with_rect_augmentation and (config.with_iq_matrices or config.with_magnitude_phase):
        config.__setattr__("model_input_dim", [126, config.rect_augment_num_of_timesteps, 2])
    return config


def main():
    # capture the config path from the run arguments
    # then process configuration file
    SRC_DIR = os.getcwd()
    RADAR_DIR = os.path.join(SRC_DIR, os.pardir)
    config = preprocess_meta_data(SRC_DIR)
    exp_name = config.exp_name

    # for graph_dir and log file
    now = datetime.now()
    date = now.strftime("%Y_%m_%d_%H_%M_%S")
    exp_name_time = '{}_{}'.format(exp_name, date)
    # visualize training performance
    graph_path = os.path.join(RADAR_DIR, 'graphs', exp_name_time)
    if os.path.exists(graph_path) is False:
        os.makedirs(graph_path)
    LOG_DIR = os.path.join(RADAR_DIR, 'logs')
    if os.path.exists(LOG_DIR) is False:
        os.makedirs(LOG_DIR)
    log_path = '{}/{}.log'.format(LOG_DIR, exp_name_time)

    config = adjust_input_size(config)

    # assert configurations
    assert not (config.learn_background and config.with_rect_augmentation)
    # assert not(config.background_implicit_inference)
    assert not (config.load_complete_model_from_file and config.load_model_weights_from_file)
    assert config.load_complete_model_from_file or config.load_model_weights_from_file

    if config.load_model_weights_from_file:
        # build the model
        print('CURRENT DIR: {}'.format(os.getcwd()))
        adjust_input_size(config)
        model_dict = build_model(config)
        model_dict['train'].load_weights(config.model_weights_file)
        model = model_dict['train']
        model.compile(optimizer=Adam(learning_rate=config.learning_rate), loss=BinaryCrossentropy(),
                      metrics=['accuracy', AUC()])
        # model_name = 'full_test_auc_95_0168'
        # print('saveing model to: {}/{}'.format(os.getcwd(),model_name))
        # model.save(model_name)
    elif config.load_complete_model_from_file:
        model = tf.keras.models.load_model(config.complete_model_file)
    else:
        raise Exception('Invalid Configuration...')

        # evaluate model
    if config.use_public_test_set:
        print(40 * '#')
        print('Model evaluation on FULL public test set:')
        os.chdir(SRC_DIR)
        eval_dataparser = DataSetParser(stable_mode=False, read_validation_only=True, config=config)
        X_valid, labels_valid = eval_dataparser.get_dataset_by_snr(dataset_type='validation', snr_type=config.snr_type)
        y_valid = np.array(labels_valid['target_type'])
        if config.with_rect_augmentation:
            X_augmented_test = expand_test_by_sampling_rect(data=X_valid, config=config)
            y_pred = []
            for sampled_list_x,test_index in zip(X_augmented_test,range(len(X_augmented_test))):
                sample_result_list = []
                sampled_list_x = np.array(sampled_list_x)
                x = np.expand_dims(sampled_list_x,axis=-1)
                sample_result_list.extend(model.predict(x,batch_size=x.shape[0]).flatten().tolist())
                y_pred.append(np.mean(sample_result_list))
            # raise Exception('Currently not supported')
            y_pred = np.array(y_pred)
        else:
            X_valid = np.expand_dims(X_valid, axis=-1)
            y_pred = model.predict(X_valid)
            res = model.evaluate(X_valid, y_valid)
        print('roc_auc_score on FULL public test: {}'.format(roc_auc_score(y_valid, y_pred)))
    else:
        raise Exception('Invalid Configuration..., use config.use_public_test_set = True')

    SUB_DIR = os.path.join(RADAR_DIR, 'submission_files')
    BEST_RESULT_DIR = os.path.join(RADAR_DIR, 'best_preformance_history')
    if os.path.exists(SUB_DIR) is False:
        os.makedirs(SUB_DIR)
    sub_path = "{}/submission_{}.csv".format(SUB_DIR, exp_name_time)
    test_model(model, sub_path, SRC_DIR, config, BEST_RESULT_DIR)

    # if config.save_history_buffer is True:

    print('#' * 70)
    print('submission file is at: {}'.format(sub_path))
    print('')


if __name__ == '__main__':
    print('Current working directory is: {}'.format(os.getcwd()))
    main()
