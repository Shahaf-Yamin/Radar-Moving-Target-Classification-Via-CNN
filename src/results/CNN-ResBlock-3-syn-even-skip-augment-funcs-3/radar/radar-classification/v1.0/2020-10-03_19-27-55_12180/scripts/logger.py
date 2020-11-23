import datetime
import logging
import os
import tensorflow as tf

def set_logger(config):
    ''' define logger object to log into file and to stdout'''

    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(message)s")

    if not config.quiet:
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

    log_path = os.path.join(config.tensor_board_dir, "logger.log")
    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    logger.propagate = False

def set_logger_and_tracker(config):
    ''' configure the mlflow tracker:
        1. set tracking location (uri)
        2. configure exp name/id
        3. define parameters to be documented
    '''
    config.tensor_board_dir = os.path.join('',
                                           'results',
                                           config.exp_name,
                                           config.data_name,
                                           config.run_name,
                                           config.tag_name,
                                           "{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                          config.seed))
    if not os.path.exists(config.tensor_board_dir):
        os.makedirs(config.tensor_board_dir)

    set_logger(config)

    train_log_dir = os.path.join(config.tensor_board_dir, 'train')
    config.train_writer = tf.summary.create_file_writer(train_log_dir)

    test_log_dir =  os.path.join(config.tensor_board_dir, 'test')
    config.test_writer = tf.summary.create_file_writer(test_log_dir)

