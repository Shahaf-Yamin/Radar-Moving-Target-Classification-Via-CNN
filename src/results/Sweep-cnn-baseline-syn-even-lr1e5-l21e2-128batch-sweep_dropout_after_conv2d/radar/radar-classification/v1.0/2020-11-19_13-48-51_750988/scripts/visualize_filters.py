from matplotlib import pyplot
from tensorflow.keras.models import Model
from data.data_parser import DataSetParser
from models.models import build_or_load_model
import numpy as np
import os
from utils.utils import read_config,get_args

def normalize_filters(filters):
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    return filters

def plot_filters(filters):
    # plot first few filters
    n_filters, ix = 6, 1
    for i in range(n_filters):
        # get the filter
        f = filters[:, :, :, i]
        # plot each channel separately
        for j in range(3):
            # specify subplot and turn of axis
            ax = pyplot.subplot(n_filters, 3, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(f[:, :, j], cmap='gray')
            ix += 1
    # show the figure
    pyplot.show()

def plot_feature_map(model,X):
    # plot all 64 maps in an 8x8 squares
    square = 8
    ix = 1
    # redefine model to output right after the first hidden layer
    outputs = []
    filters = []
    layer_index = 0
    print('Starting to extract layers outputs')
    for layer in model.layers:
        if 'conv' not in layer.name:
            layer_index += 1
            continue
        filter, biases = layer.get_weights()
        filters.append(normalize_filters(filter))
        outputs.append(model.layers[layer_index+1].output)# Leaky relU output
        layer_index += 1
        if layer_index > 10:
            break

    model = Model(inputs=model.inputs, outputs=outputs)
    print('Predicting features')
    feature_maps = model.predict(X)
    # plot the output from each block
    square = 4
    print('Printing features')
    if not os.path.exists('visualization'):
        os.mkdir(f'visualization')
    pyplot.imshow(X[:, :, 0])
    pyplot.savefig('visualization/Original Sample.png')
    print(f'Original Image saved to {os.getcwd()}/visualization')
    for fmap,layer_index,filter in zip(feature_maps,range(len(feature_maps)),filters):
        # plot all 64 maps in an 8x8 squares
        ix = 1
        print(fmap.shape)
        if not os.path.exists(f'visualization/Layer_{layer_index}'):
            os.mkdir(f'visualization/Layer_{layer_index}')

        for filter_index in range(fmap.shape[-1]):
            pyplot.figure()
            pyplot.imshow(fmap[:, :, 0, filter_index])
            pyplot.savefig(f'visualization/Layer_{layer_index}/{filter_index}_feature_map.png')

        pyplot.figure()
        for _ in range(4):
            for _ in range(8):
                # specify subplot and turn of axis
                ax = pyplot.subplot(4, 8, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                pyplot.imshow(fmap[:, :, 0, ix - 1])
                ix += 1
        # show the figure
        pyplot.savefig(f'visualization/Layer_{layer_index}/subplot_32_feature_maps.png')
        pyplot.figure()
        ix = 1
        for j in range(32):
            # specify subplot and turn of axis
            ax = pyplot.subplot(4, 8, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            print(filter.shape)
            pyplot.imshow(filter[:,:,0,j])
            ix += 1
        pyplot.savefig(f'visualization/Layer_{layer_index}/subplot_32_filters.png')

def visualize_filters(model,config):
    stable_mode = config.get('stable_mode')
    testset_size = config.get('N_test')
    print('Loading data set')
    data_parser = DataSetParser(stable_mode=stable_mode, config=config)
    X = data_parser.train_data[0][0]
    print(X.shape)
    X = np.expand_dims(X, axis=-1)
    print(X.shape)
    plot_feature_map(model, X)

if __name__ == '__main__':
    print('CURRENT DIR: {}'.format(os.getcwd()))
    args = get_args()
    config = read_config(args)
    model = build_or_load_model(config)
    visualize_filters(model=model['train'], config=config)
