#!/usr/bin/env python3

from trainers.trainer import build_trainer
from data.data_loader import load_data
from models.models import build_model
from utils.utils import preprocess_meta_data
from data.data_parser import DataSetParser
from pathlib import Path
from utils.result_utils import analyse_model_performance
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

matplotlib.use('Agg')

tf.keras.backend.set_floatx('float32')


def test_model(model, sub_path, SRC_DIR,config):
    os.chdir(SRC_DIR)
    test_dataloader = DataSetParser(stable_mode=False, read_test_only=True, config=config)

    X_test = test_dataloader.get_dataset_test_allsnr()

    # swap axes for sequential data
    if bool(re.search('LSTM',config.exp_name,re.IGNORECASE)) or bool(re.search('tcn',config.exp_name,re.IGNORECASE)):
        X_test = X_test.swapaxes(1, 2)
    else:
        X_test = np.expand_dims(X_test, axis=-1)

    # Creating DataFrame with the probability prediction for each segment
    submission = pd.DataFrame()
    submission['segment_id'] = test_dataloader.test_data[1]['segment_id']
    submission['prediction'] = model.predict(X_test).flatten().tolist()
    submission['prediction'] = submission['prediction'].astype('float')
    # Save submission
    submission.to_csv(sub_path, index=False)

def test_model_ensemble(model, sub_path, SRC_DIR,config):
    os.chdir(SRC_DIR)
    test_dataloader = DataSetParser(stable_mode=False, read_test_only=True, config=config)

    X_test = test_dataloader.get_dataset_test_allsnr()

    # swap axes for sequential data
    if bool(re.search('LSTM',config.exp_name,re.IGNORECASE)) or bool(re.search('tcn',config.exp_name,re.IGNORECASE)):
        X_test = X_test.swapaxes(1, 2)
    else:
        X_test = np.expand_dims(X_test, axis=-1)

    # Creating DataFrame with the probability prediction for each segment
    submission = pd.DataFrame()
    submission['segment_id'] = test_dataloader.test_data[1]['segment_id']

    for x,snr_type in zip(X_test,test_dataloader.test_data[1]['snr_type']):
        if snr_type == 'LowSNR':
            submission['prediction'].append(model['LowSNR'].predict(X_test).flatten())
        if snr_type == 'HighSNR':
            submission['prediction'].append(model['HighSNR'].predict(X_test).flatten())

    submission['prediction'] = submission['prediction'].astype('float')
    # Save submission
    submission.to_csv(sub_path, index=False)


def main():
    # capture the config path from the run arguments
    # then process configuration file
    SRC_DIR = os.getcwd()
    RADAR_DIR = os.path.join(SRC_DIR, os.pardir)
    config = preprocess_meta_data()
    exp_name = config.exp_name

    # for graph_dir and log file
    now = datetime.now()
    date = now.strftime("%Y_%m_%d_%H_%M_%S")
    exp_name_time = '{}_{}'.format(exp_name, date)
    # visualize training performance
    graph_path = os.path.join(RADAR_DIR, 'graphs', exp_name_time)
    LOG_DIR = os.path.join(RADAR_DIR, 'logs')
    if os.path.exists(LOG_DIR) is False:
        os.makedirs(LOG_DIR)
    log_path = '{}/{}.log'.format(LOG_DIR, exp_name_time)

    if config.use_mti_improvement is True:
        config.__setattr__("model_input_dim", [125, 32, 1])

    if bool(re.search('tcn',config.exp_name,re.IGNORECASE)) and config.use_mti_improvement:
        config.__setattr__("model_input_dim", [32, 125, 1])

    # load the data
    data = load_data(config)

    # create a model
    model = build_model(config)

    # create trainer
    trainer = build_trainer(model, data, config)

    # train the model
    history = trainer.train()

    analyse_model_performance(model, data, history, config, graph_path=graph_path, res_dir=exp_name_time)

    # evaluate model
    eval_res = trainer.evaluate()


    SUB_DIR = os.path.join(RADAR_DIR,'submission_files')
    if os.path.exists(SUB_DIR) is False:
        os.makedirs(SUB_DIR)
    sub_path = "{}/submission_{}.csv".format(SUB_DIR,exp_name_time)
    test_model(model['train'], sub_path, SRC_DIR, config)


    print('#' * 70)
    print('log file is located at {}'.format(log_path))
    print('graphs are located at {}'.format(graph_path))
    print('submission file is at: {}'.format(sub_path))
    print('')


if __name__ == '__main__':
    print('Current working directory is: {}'.format(os.getcwd()))
    main()
