from trainers.trainer import build_trainer
from data.data_loader import load_data
from models.models import build_model
from utils.utils import preprocess_meta_data
from data.data_parser import DataSetParser
from pathlib import Path
from utils.result_utils import analyse_model_performance, print_sweep_by_parameter
import pandas as pd
import tensorflow as tf
import os
import datetime
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from utils.utils import Unbuffered
import sys
from collections import OrderedDict
import re
from datetime import datetime


def sweep_core(config, graph_path, res_dir):
    # load the data
    data = load_data(config)

    # create a model
    model = build_model(config)

    # create trainer
    trainer = build_trainer(model, data, config)

    # train the model
    history = trainer.train()

    analyse_model_performance(model, data, history, config, graph_path=graph_path, res_dir=res_dir)

    # evaluate model
    eval_res = trainer.evaluate()
    model = trainer.model_train
    res_dict = OrderedDict()
    for key in history.history.keys():
        res_dict[key] = history.history[key][-1]
    return res_dict, history.history


def main_sweep():
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

    sys.stdout = Unbuffered(stream=sys.stdout, path=log_path)

    sweep_list = config.params_to_sweep
    res_dict = OrderedDict()
    hist_dict = OrderedDict()
    orig_config = config
    for list_name in sweep_list:
        param_name = re.sub('_list', '', list_name)
        res_dict[param_name] = {}
        hist_dict[param_name] = {}
        for param in config.get(list_name):
            config = orig_config
            config.__setattr__(param_name, param)
            res_dir = '{}_{}'.format(param_name, param)
            print('#' * 70)
            print('Sweeping parameter: {}, with value: {}'.format(param_name, param))
            if type(param) == list:
                param = '_'.join(param)
            res_dict[param_name][param], hist_dict[param_name][param] = sweep_core(config, graph_path=graph_path, res_dir=res_dir)

    print('')
    print('#' * 70)
    print('Sweep results summary')
    print('#' * 70)
    print('log file is located at {}'.format(log_path))
    print('graphs are located at {}'.format(graph_path))
    print('')
    for param_name in res_dict.keys():
        for param in res_dict[param_name].keys():
            print('{} = {}'.format(param_name, param))
            for metric in res_dict[param_name][param].keys():
                print('{}: {}, '.format(metric, res_dict[param_name][param][metric]), end='')
            print('')



    for list_name in sweep_list:
        param_name = re.sub('_list', '', list_name)
        print_sweep_by_parameter(hist_dict, param_name=param_name, metric_list=['val_accuracy', 'accuracy'],
                             graph_path=graph_path, title='{} Accuracy'.format(param_name))
        print_sweep_by_parameter(hist_dict, param_name=param_name, metric_list=['val_auc', 'auc'],
                             graph_path=graph_path, title='{} AUC'.format(param_name))


if __name__ == '__main__':
    print('Current working directory is: {}'.format(os.getcwd()))
    main_sweep()
