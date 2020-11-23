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
from matplotlib import pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('Agg')

tf.keras.backend.set_floatx('float32')


def test_model(model, SRC_DIR, config):
    os.chdir(SRC_DIR)
    test_dataloader = DataSetParser(stable_mode=False, read_test_only=True)
    src_path = Path(SRC_DIR)
    parent_path = str(src_path.parent)
    os.chdir(parent_path)
    X_test = test_dataloader.get_dataset_test_allsnr()
    # swap axes for sequential data
    if config.exp_name == "LSTM":
        X_test = X_test.swapaxes(1, 2)
    else:
        X_test = np.expand_dims(X_test, axis=-1)

    # Creating DataFrame with the probability prediction for each segment
    submission = pd.DataFrame()
    submission['segment_id'] = test_dataloader.test_data[1]['segment_id']
    submission['prediction'] = model.predict(X_test).flatten().tolist()
    submission['prediction'] = submission['prediction'].astype('float')
    # Save submission
    submission.to_csv('submission.csv', index=False)




def main():
    # capture the config path from the run arguments
    # then process configuration file
    SRC_DIR = os.getcwd()
    config = preprocess_meta_data()

    # load the data
    data = load_data(config)

    if not config.quiet:
        config.print()

    # create a model
    model = build_model(config)

    # create trainer
    trainer = build_trainer(model, data, config)

    # train the model
    history = trainer.train()

    # visualize training performance
    graph_path = os.path.join(SRC_DIR, 'graphs')

    analyse_model_performance(model, data, graph_path, history)

    # evaluate model
    trainer.evaluate()

    # run on MAFAT test
    model = trainer.model_train
    test_model(model, SRC_DIR, config)



if __name__ == '__main__':
    print('Current working directory is: {}'.format(os.getcwd()))
    main()
