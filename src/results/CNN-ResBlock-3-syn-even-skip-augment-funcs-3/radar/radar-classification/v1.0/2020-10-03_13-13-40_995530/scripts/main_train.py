#!/usr/bin/env python3

from trainers.trainer import build_trainer
from data.data_loader import load_data,expand_test_by_sampling_rect
from models.models import build_model
from utils.utils import preprocess_meta_data
from data.data_parser import DataSetParser
from data.signal_processing import Sample_rectangle_from_spectrogram
from pathlib import Path
from utils.result_utils import analyse_model_performance,compare_to_best_model_performance,save_model
import pandas as pd
import tensorflow as tf
import os
import datetime
import matplotlib
import numpy as np
from utils.utils import Unbuffered
import sys
from datetime import datetime
import re
from copy import deepcopy

matplotlib.use('Agg')

tf.keras.backend.set_floatx('float32')


def test_model(model, sub_path, SRC_DIR,config,BEST_RESULT_DIR):
    os.chdir(SRC_DIR)
    test_dataloader = DataSetParser(stable_mode=False, read_test_only=True, config=config)
    X_test = test_dataloader.get_dataset_test_allsnr()
    if config.with_rect_augmentation or config.with_preprocess_rect_augmentation:
        X_augmented_test = expand_test_by_sampling_rect(data=X_test,config=config)
    # swap axes for sequential data
    elif bool(re.search('LSTM',config.exp_name,re.IGNORECASE)) or bool(re.search('tcn',config.exp_name,re.IGNORECASE)):
        X_test = X_test.swapaxes(1, 2)
    else:
        X_test = np.expand_dims(X_test, axis=-1)

    result_list = []
    segment_list = []
    result_list_temp = []
    submission = pd.DataFrame()

    # Creating DataFrame with the probability prediction for each segment
    if config.snr_type == 'all':
        segment_list = test_dataloader.test_data[1]['segment_id']
        if config.with_rect_augmentation or config.with_preprocess_rect_augmentation:
            for sampled_list_x,test_index in zip(X_augmented_test,range(len(X_augmented_test))):
                sample_result_list = []
                prev_gap_doppler_burst = config.rect_augment_gap_doppler_burst_from_edge
                while not sampled_list_x:
                    '''
                    That means that we didn't manged to sample rectangle with the current doppler burst gap
                    '''
                    config.rect_augment_gap_doppler_burst_from_edge -= 1
                    print('Reducing the doppler burst gap for test_index sample {} '.format(test_index))
                    sampled_list_x = Sample_rectangle_from_spectrogram(X_test[test_index],config)

                config.rect_augment_gap_doppler_burst_from_edge = prev_gap_doppler_burst
                print('Sampled {} rectangles for test_index sample {} '.format(len(sampled_list_x), test_index))
                sampled_list_x = np.array(sampled_list_x)
                x = np.expand_dims(sampled_list_x,axis=-1)
                sample_result_list.extend(model.predict(x,batch_size=x.shape[0]).flatten().tolist())
                # result_list.append(np.mean(sample_result_list))
                result_list_temp.append(np.mean(sample_result_list))
        else:
            # result_list = model.predict(X_test).flatten().tolist()
            result_list_temp = model.predict(X_test).flatten().tolist()
    elif config.snr_type == 'low':
        if config.with_rect_augmentation or config.with_preprocess_rect_augmentation:
            for sampled_list_x, snr_type, segment_id,test_index in zip(X_augmented_test, test_dataloader.test_data[1]['snr_type'],
                                                                      test_dataloader.test_data[1]['segment_id'],range(len(X_augmented_test))):
                if snr_type == 'LowSNR':
                    sample_result_list = []
                    prev_gap_doppler_burst = config.rect_augment_gap_doppler_burst_from_edge
                    while not sampled_list_x:
                        '''
                        That means that we didn't manged to sample rectangle with the current doppler burst gap
                        '''
                        config.rect_augment_gap_doppler_burst_from_edge -= 1
                        print('Reducing the doppler burst gap for test_index sample {} '.format(test_index))
                        sampled_list_x = Sample_rectangle_from_spectrogram(X_test[test_index], config)

                    config.rect_augment_gap_doppler_burst_from_edge = prev_gap_doppler_burst
                    print('Sampled {} rectangles for test_index sample {} '.format(len(sampled_list_x), test_index))
                    sampled_list_x = np.array(sampled_list_x)
                    x = np.expand_dims(sampled_list_x, axis=-1)
                    sample_result_list.extend(model.predict(x, batch_size=x.shape[0]).flatten().tolist())
                    # result_list.append(np.mean(sample_result_list))
                    result_list_temp.append(np.mean(sample_result_list))
        else:
            low_snr_list = []
            for x,snr_type,segment_id in zip(X_test,test_dataloader.test_data[1]['snr_type'],test_dataloader.test_data[1]['segment_id']):
                if snr_type == 'LowSNR':
                    low_snr_list.append(x)
                    segment_list.append(segment_id)
            sampled_list_x = np.array(low_snr_list)
            x = np.expand_dims(sampled_list_x, axis=-1)
            # result_list = model.predict(x, batch_size=x.shape[0]).flatten().tolist()
            result_list_temp = model.predict(x, batch_size=x.shape[0]).flatten().tolist()
    else:
        # High SNR run
        if config.with_rect_augmentation or config.with_preprocess_rect_augmentation:
            for sampled_list_x, snr_type, segment_id,test_index in zip(X_augmented_test, test_dataloader.test_data[1]['snr_type'],
                                                           test_dataloader.test_data[1]['segment_id'],range(len(X_augmented_test))):
                if snr_type == 'HighSNR':
                    sample_result_list = []
                    prev_gap_doppler_burst = config.rect_augment_gap_doppler_burst_from_edge
                    while not sampled_list_x:
                        '''
                        That means that we didn't manged to sample rectangle with the current doppler burst gap
                        '''
                        config.rect_augment_gap_doppler_burst_from_edge -= 1
                        print('Reducing the doppler burst gap for test_index sample {} '.format(test_index))
                        sampled_list_x = Sample_rectangle_from_spectrogram(X_test[test_index], config)

                    config.rect_augment_gap_doppler_burst_from_edge = prev_gap_doppler_burst
                    print('Sampled {} rectangles for test_index sample {} '.format(len(sampled_list_x), test_index))
                    sampled_list_x = np.array(sampled_list_x)
                    x = np.expand_dims(sampled_list_x, axis=-1)
                    sample_result_list.extend(model.predict(x, batch_size=x.shape[0]).flatten().tolist())
                    # result_list.append(np.mean(sample_result_list))
                    result_list_temp.append(np.mean(sample_result_list))
        else:
            high_snr_list = []
            for x,snr_type,segment_id in zip(X_test,test_dataloader.test_data[1]['snr_type'],test_dataloader.test_data[1]['segment_id']):
                if snr_type == 'HighSNR':
                    high_snr_list.append(x)
                    segment_list.append(segment_id)
            sampled_list_x = np.array(high_snr_list)
            x = np.expand_dims(sampled_list_x, axis=-1)
            # result_list = model.predict(x, batch_size=x.shape[0]).flatten().tolist()
            result_list_temp = model.predict(x, batch_size=x.shape[0]).flatten().tolist()


    if config.learn_background:
        result_list_temp = np.array(result_list_temp).reshape((-1, 3))
        if config.background_implicit_inference:
            y_pred_2 = np.array([[y[0] , y[1] + y[2]] for y in result_list_temp])
        else:
            y_pred_2 = np.array([[y[0] / (1 - y[2]), y[1] / (1 - y[2])] for y in result_list_temp])
        y_pred_2 = np.array([y / (y[0] + y[1]) if y[0] + y[1] > 1 else y for y in y_pred_2]) # numeric correction
        result_list = [y[0] if y[0] > y[1] else 1 - y[1] for y in y_pred_2]
    else:
        result_list = result_list_temp

    submission['segment_id'] = segment_list
    submission['prediction'] = result_list
    submission['prediction'] = submission['prediction'].astype('float')
    # Save submission
    submission.to_csv(sub_path, index=False)
def adjust_input_size(config):
    if config.with_iq_matrices is True or config.with_magnitude_phase is True:
        config.__setattr__("model_input_dim", [126, 32, 2])
    if config.with_rect_augmentation or config.with_preprocess_rect_augmentation:
        config.__setattr__("model_input_dim", [config.rect_augment_num_of_rows,config.rect_augment_num_of_cols, 1])
    elif (config.with_rect_augmentation or config.with_preprocess_rect_augmentation) and config.with_iq_matrices:
        config.__setattr__("model_input_dim", [config.rect_augment_num_of_rows,config.rect_augment_num_of_cols, 2])
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

    '''
    Configure multiprocess
    '''
    strategy = tf.distribute.MirroredStrategy()

    if strategy.num_replicas_in_sync != 1:
        config.__setattr__("batch_size", config.batch_size * strategy.num_replicas_in_sync)

    config = adjust_input_size(config)

    # assert configurations
    assert not(config.learn_background and (config.with_rect_augmentation or config.with_preprocess_rect_augmentation))
    # assert not(config.background_implicit_inference)
    # load the data
    data = load_data(config)

    with strategy.scope():
        # create a model
        model = build_model(config)

        # create trainer
        trainer = build_trainer(model, data, config)

        # train the model
        history = trainer.train()

    # evaluate model
    eval_res = trainer.evaluate()

    SUB_DIR = os.path.join(RADAR_DIR,'submission_files')
    BEST_RESULT_DIR = os.path.join(RADAR_DIR, 'best_preformance_history')
    if os.path.exists(SUB_DIR) is False:
        os.makedirs(SUB_DIR)
    sub_path = "{}/submission_{}.csv".format(SUB_DIR,exp_name_time)
    test_model(model['train'], sub_path, SRC_DIR, config,BEST_RESULT_DIR)


    if config.learn_background is False:
        result_data = analyse_model_performance(model, data, history, config, graph_path=graph_path, res_dir=exp_name_time)
        result_data['Log path'] = log_path
        result_data['Graph path'] = graph_path
        result_data['Submission path'] = sub_path
        result_data['Model name'] = config.model_name
        result_data['Exp name'] = config.exp_name
        result_data['Snr type'] = config.snr_type

        # compare model performance
        if os.path.exists(BEST_RESULT_DIR) is False:
            os.makedirs(BEST_RESULT_DIR)

        compare_to_best_model_performance(result_data, model, BEST_RESULT_DIR, config)

    PREVIOUS_MODELS_DIR = os.path.join(RADAR_DIR, 'previous_models_files')
    if config.save_model is True:
        if os.path.exists(PREVIOUS_MODELS_DIR) is False:
            os.makedirs(PREVIOUS_MODELS_DIR)
        os.chdir(PREVIOUS_MODELS_DIR)
        save_model(name='{}_{}_{}'.format(config.model_name,config.exp_name,exp_name_time), model=model['train'])

    #if config.save_history_buffer is True:

    print('#' * 70)
    print('log file is located at {}'.format(log_path))
    print('graphs are located at {}'.format(graph_path))
    print('submission file is at: {}'.format(sub_path))
    print('')


if __name__ == '__main__':
    print('Current working directory is: {}'.format(os.getcwd()))
    main()
