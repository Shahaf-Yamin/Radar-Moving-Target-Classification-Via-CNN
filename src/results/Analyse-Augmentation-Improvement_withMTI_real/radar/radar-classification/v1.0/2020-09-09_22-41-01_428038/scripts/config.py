from bunch import Bunch
from collections import OrderedDict
import json
import logging
from pathlib import Path
from random import randint

CONFIG_VERBOSE_WAIVER = ['save_model', 'tracking_uri', 'quiet', 'sim_dir', 'train_writer', 'test_writer', 'valid_writer']
MAX_SEED = 1000000
logger = logging.getLogger("logger")

class Config(Bunch):
    """ class for handling dicrionary as class attributes """

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)

    def print(self):
        line_len = 122
        line = "-" * line_len
        logger.info(line + "\n" +
              "| {:^35s} | {:^80} |\n".format('Feature', 'Value') +
              "=" * line_len)
        for key, val in sorted(self.items(), key= lambda x: x[0]):
            if isinstance(val, OrderedDict):
                raise NotImplementedError("Nested configs are not implemented")
            else:
                if key not in CONFIG_VERBOSE_WAIVER:
                    logger.info("| {:35s} | {:80} |\n".format(key, str(val)) + line)
        logger.info("\n")

def read_json_to_dict(fname):
    """ read json config file into ordered-dict """
    fname = Path(fname)
    with fname.open('rt') as handle:
        config_dict = json.load(handle, object_hook=OrderedDict)
        return config_dict

def read_config(args):
    """ read config from json file and update by the command line arguments """
    if args.config is not None:
        json_file = args.config
    else:
        raise ValueError("preprocess config: config path wasn't specified")

    config_dict = read_json_to_dict(json_file)
    config = Config(config_dict)

    for arg in sorted(vars(args)):
        key = arg
        val = getattr(args, arg)
        if val is not None:
            setattr(config, key, val)

    if args.seed is None and config.seed is None:
        config.seed = randint(0, MAX_SEED)

    return config

