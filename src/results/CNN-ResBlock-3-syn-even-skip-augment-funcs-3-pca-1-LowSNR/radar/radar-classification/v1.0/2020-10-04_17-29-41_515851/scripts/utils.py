import argparse
import logging
import os
import glob
import shutil
import sys
import tensorflow as tf
from utils.config import read_config
from utils.logger import set_logger_and_tracker
from datetime import datetime

logger = logging.getLogger("logger")


def get_args():
    """" collects command line arguments """

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--config', default=None, type=str, help='path to config file')
    argparser.add_argument('--exp_name', default=None, type=str, help='experiment name')
    argparser.add_argument('--run_name', default=None, type=str, help='run name')
    argparser.add_argument('--tag_name', default=None, type=str, help='tag name')
    argparser.add_argument('--batch_size', default=None, type=int, help='batch size in training')
    argparser.add_argument('--seed', default=None, type=int, help='randomization seed')
    argparser.add_argument('--N_test', default=None, type=int,
                           help='Number of test examples held out from the train set')
    argparser.add_argument('--stable_mode', default=None, type=bool,
                           help='if 0 load only test and train, if 1 load full dataset')
    argparser.add_argument('--quiet', dest='quiet', action='store_true')
    argparser.set_defaults(quiet=False)

    args = argparser.parse_args()
    return args


def gpu_init():
    """ Allows GPU memory growth """

    gpus = tf.config.experimental.list_physical_devices('GPU')
    logger.info("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            logger.info(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logger.info("MESSAGE", e)


def save_scripts(config,SRC_DIR):
    path = os.path.join(config.tensor_board_dir, 'scripts')
    if not os.path.exists(path):
        os.makedirs(path)
    scripts_to_save = glob.glob('{}/**/*.py'.format(SRC_DIR), recursive=True) + [config.config]
    scripts_to_save = [script for script in scripts_to_save if '{}/results'.format(SRC_DIR) not in script]
    if scripts_to_save is not None:
        for script in scripts_to_save:
            dst_file = os.path.join(path, os.path.basename(script))
            shutil.copyfile(os.path.join(os.path.dirname(sys.argv[0]), script), dst_file)


def preprocess_meta_data(SRC_DIR):
    """ preprocess the config for specific run:
            1. reads command line arguments
            2. updates the config file and set gpu config
            3. configure gpu settings
            4. Define logger
            5. Save scripts
    """

    args = get_args()

    config = read_config(args)

    gpu_init()

    set_logger_and_tracker(config)

    save_scripts(config,SRC_DIR)

    print_config(config)

    return config


def print_config(config):
    print('')
    print('#' * 70)
    print('Configurations at beginning of run')
    print('#' * 70)
    for key in config.keys():
        print('{}, {}'.format(key,config['{}'.format(key)]))
    print('')
    print('')


class Unbuffered:
    def __init__(self, stream, path):
        print('CURRENT IS : {}'.format(os.getcwd()))
        self.log = open(path, "a")

        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.log.write(data)  # Write the data of stdout here to a text file as well

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
