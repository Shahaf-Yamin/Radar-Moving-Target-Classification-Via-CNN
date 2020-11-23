#!/usr/bin/env python3

from models.models import build_model
from utils.utils import preprocess_meta_data
from main_train import test_model
import tensorflow as tf
import os
import datetime
import matplotlib
from datetime import datetime

matplotlib.use('Agg')

tf.keras.backend.set_floatx('float32')


def adjust_input_size(config):
    if config.with_iq_matrices is True or config.with_magnitude_phase is True:
        config.__setattr__("model_input_dim", [126, 32, 2])
    if config.with_rect_augmentation:
        config.__setattr__("model_input_dim", [126,config.rect_augment_num_of_timesteps, 1])
    if config.with_rect_augmentation and (config.with_iq_matrices or config.with_magnitude_phase):
        config.__setattr__("model_input_dim", [126,config.rect_augment_num_of_timesteps, 2])
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
    assert not(config.learn_background and config.with_rect_augmentation)
    # assert not(config.background_implicit_inference)
    assert not(config.load_complete_model_from_file and config.load_model_weights_from_file)
    assert config.load_complete_model_from_file or config.load_model_weights_from_file

    if config.load_model_weights_from_file:
        # build the model
        print('CURRENT DIR: {}'.format(os.getcwd()))
        model_dict = build_model(config)
        model_dict['train'].load_weights(config.model_weights_file)
        model = model_dict['train']
    elif config.load_complete_model_from_file:
        model = tf.keras.models.load_model(config.complete_model_file)
    else:
        raise Exception('Invalid Configuration...')


    SUB_DIR = os.path.join(RADAR_DIR,'submission_files')
    BEST_RESULT_DIR = os.path.join(RADAR_DIR, 'best_preformance_history')
    if os.path.exists(SUB_DIR) is False:
        os.makedirs(SUB_DIR)
    sub_path = "{}/submission_{}.csv".format(SUB_DIR,exp_name_time)
    test_model(model, sub_path, SRC_DIR, config,BEST_RESULT_DIR)


    #if config.save_history_buffer is True:

    print('#' * 70)
    print('submission file is at: {}'.format(sub_path))
    print('')


if __name__ == '__main__':
    print('Current working directory is: {}'.format(os.getcwd()))
    main()
