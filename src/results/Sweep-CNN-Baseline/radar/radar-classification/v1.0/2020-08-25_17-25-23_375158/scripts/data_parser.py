import pickle
import os
import csv
import collections
import numpy as np
from os.path import expanduser
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib

class DataSetParser(object):
    def __init__(self, stable_mode, read_test_only=False):
        self.radar_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
        self.dataset_dir = '{}/dataset'.format(self.radar_dir)
        self.dataset_pickle = os.listdir(self.dataset_dir)
        self.aux_exp_file_path = "MAFAT RADAR Challenge - Auxiliary Experiment Set V2"

        if read_test_only is True:
            self.test_data = self.dump_data(pickle_name='MAFAT RADAR Challenge - Public Test Set V1.pkl',
                                            csv_name='MAFAT RADAR Challenge - Public Test Set V1.csv')
            return
        # load train
        self.train_data = self.dump_data(pickle_name='MAFAT RADAR Challenge - Training Set V1.pkl',
                                         csv_name='MAFAT RADAR Challenge - Training Set V1.csv')
        if stable_mode:
            # self.aux_syn_data = self.dump_data(pickle_name='MAFAT RADAR Challenge - Auxiliary Synthetic Set V2.pkl',
            #                                    csv_name='MAFAT RADAR Challenge - Auxiliary Synthetic Set V2.csv')
            # self.aux_exp_data = self.dump_data(pickle_name='MAFAT RADAR Challenge - Auxiliary Experiment Set V2.pkl',
            #                                    csv_name='MAFAT RADAR Challenge - Auxiliary Experiment Set V2.csv')
            self.aux_exp_data = self.load_data_aux_exp_data()
            # self.aux_background_data = self.dump_data(
            #     pickle_name='MAFAT RADAR Challenge - Auxiliary Background(empty) Set V1.pkl',
            #     csv_name='MAFAT RADAR Challenge - Auxiliary Background(empty) Set V1.csv')

    def fft(self, iq, axis=0, window=None):
        """
          Computes the log of discrete Fourier Transform (DFT).

          Arguments:
            iq_burst -- {ndarray} -- 'iq_sweep_burst' array
            axis -- {int} -- axis to perform fft in (Default = 0)

          Returns:
            log of DFT on iq_burst array
        """
        smooth_iq = self.hann(iq, window)
        iq = np.log(np.abs(np.fft.fft(smooth_iq, axis=axis)))
        return iq

    def max_value_on_doppler(self, iq, doppler_burst):
        """
        Set max value on I/Q matrix using doppler burst vector.

        Arguments:
          iq_burst -- {ndarray} -- 'iq_sweep_burst' array
          doppler_burst -- {ndarray} -- 'doppler_burst' array (center of mass)

        Returns:
          I/Q matrix with the max value instead of the original values
          The doppler burst marks the matrix values to change by max value
        """
        iq_max_value = np.max(iq)
        for i in range(iq.shape[1]):
            if doppler_burst[i] >= len(iq):
                continue
            iq[doppler_burst[i], i] = iq_max_value
        return iq

    def normalize(self, iq):
        """
        Calculates normalized values for iq_sweep_burst matrix:
        (vlaue-mean)/std.
        """
        mean = iq.mean()
        std = iq.std()
        return (iq - mean) / std

    def hann(self, iq, window=None):
        """
        Preformes Hann smoothing of 'iq_sweep_burst'.

        Arguments:
          iq {ndarray} -- 'iq_sweep_burst' array
          window -{range} -- range of hann window indices (Default=None)
                   if None the whole column is taken

        Returns:
          Regulazied iq in shape - (window[1] - window[0] - 2, iq.shape[1])
        """
        if window is None:
            window = [0, len(iq)]

        N = window[1] - window[0] - 1
        n = np.arange(window[0], window[1])
        n = n.reshape(len(n), 1)
        hannCol = 0.5 * (1 - np.cos(2 * np.pi * (n / N)))
        return (hannCol * iq[window[0]:window[1]])[1:-1]

    def aux_split(self,segments_per_aux_track):
        """
        Selects segments from the auxilary set for training set.
        Takes the first 3 segments (or less) from each track.

        Arguments:
          self.aux_exp_data -- the auxilary data

        Returns:
          The auxilary data for the training
        """
        data = self.aux_exp_data
        idx = np.bool_(np.zeros(len(data['track_id'])))
        for track in np.unique(data['track_id']):
            idx |= data['segment_id'] == (data['segment_id'][data['track_id'] == track][:segments_per_aux_track])

        for key in data:
            data[key] = data[key][idx]

        pkl_raw_data = {'segment_id': data['segment_id'], 'iq_sweep_burst': data['iq_sweep_burst'],
                        'doppler_burst': data['doppler_burst']}
        iq = self.preprocess_pkl_data(pkl_raw_data=pkl_raw_data)
        labels_dict = collections.OrderedDict()
        key_list = ['segment_id', 'track_id', 'geolocation_type', 'geolocation_id', 'sensor_id', 'snr_type',
                    'date_index', 'target_type']
        for key in key_list:
            labels_dict[key] = data[key]
        labels_dict['target_type'] = [0 if y == 'animal' else 1 for y in labels_dict['target_type']]

        return iq, labels_dict

    def load_data_aux_exp_data(self):
        """
        Reads all data files (metadata and signal matrix data) as python dictionary,
        the pkl and csv files must have the same file name.

        Arguments:
          file_path -- {str} -- path to the iq_matrix file and metadata file

        Returns:
          Python dictionary
        """
        file_path = self.aux_exp_file_path
        pkl = self.load_pkl_data()
        meta = self.load_csv_metadata()
        data_dictionary = {**meta, **pkl}

        for key in data_dictionary.keys():
            data_dictionary[key] = np.array(data_dictionary[key])

        return data_dictionary

    def load_pkl_data(self):
        """
        Reads pickle file as a python dictionary (only Signal data).

        Arguments:
          file_path -- {str} -- path to pickle iq_matrix file

        Returns:
          Python dictionary
        """
        file_path = self.aux_exp_file_path
        # path = os.path.join(mount_path, competition_path, file_path + '.pkl')
        with open('{}.pkl'.format(file_path), 'rb') as data:
            output = pickle.load(data)
        return output

    def load_csv_metadata(self):
        """
        Reads csv as pandas DataFrame (only Metadata).

        Arguments:
          file_path -- {str} -- path to csv metadata file

        Returns:
          Pandas DataFarme
        """
        file_path = self.aux_exp_file_path
        # path = os.path.join(mount_path, competition_path, file_path + '.csv')
        with open('{}.csv'.format(file_path), 'rb') as data:
            output = pd.read_csv(data)
        return output

    # TODO: add here another windowing methods

    def preprocess_pkl_data(self, pkl_raw_data):
        doppler_burst = pkl_raw_data['doppler_burst']
        raw_iq_matrices = pkl_raw_data['iq_sweep_burst']
        iq = np.zeros((raw_iq_matrices.shape[0], raw_iq_matrices.shape[1] - 2, raw_iq_matrices.shape[2]))
        for segment in range(raw_iq_matrices.shape[0]):
            iq[segment] = self.fft(raw_iq_matrices[segment])
            iq[segment] = self.max_value_on_doppler(iq=iq[segment], doppler_burst=doppler_burst[segment])
            iq[segment] = self.normalize(iq=iq[segment])
        return iq

    def preprocess_csv_data(self, csv_labels):
        is_first_line = True
        labels_dict = collections.OrderedDict()
        for line in csv_labels:
            if is_first_line:
                is_first_line = False
                for key in line:
                    labels_dict[key] = []
            else:
                for key, data in zip(labels_dict.keys(), line):
                    labels_dict[key].append(data)
        return labels_dict

    def dump_data(self, pickle_name, csv_name):
        os.chdir(self.dataset_dir)
        with open('{}'.format(pickle_name), 'rb') as pickle_file:
            raw_data = pickle.load(pickle_file)
            iq = self.preprocess_pkl_data(pkl_raw_data=raw_data)

        with open('{}'.format(csv_name)) as csv_file:
            labels = csv.reader(csv_file, delimiter=',', quotechar='|')
            labels_dict = self.preprocess_csv_data(csv_labels=labels)

        return iq, labels_dict

    def get_dataset(self, is_high_snr=True, dataset_type='train'):
        dataset = self.choose_dataset_type(dataset_type)
        # train
        if is_high_snr:
            high_train_indices = np.where(np.asarray(dataset[1]['snr_type']) == 'HighSNR')[0].tolist()
            X_train = dataset[0][high_train_indices]

            labels = np.asarray(dataset[1]['target_type'])[high_train_indices]
            y_train = np.asarray([0 if y == 'animal' else 1 for y in y_train])
        else:
            low_train_indices = np.where(np.asarray(dataset[1]['snr_type']) != 'HighSNR')[0].tolist()
            X_train = dataset[0][low_train_indices]
            y_train = np.asarray(dataset[1]['target_type'])[low_train_indices]
            y_train = np.asarray([0 if y == 'animal' else 1 for y in y_train])

        # permutate datasaet
        perm_indices = np.random.permutation(X_train.shape[0]).tolist()
        return X_train[perm_indices], y_train[perm_indices]

    def choose_dataset_type(self, dataset_type):
        if dataset_type is 'train':
            return self.train_data
        elif dataset_type is 'test':
            return self.test_data
        elif dataset_type is 'aux_exp':
            try:
                return self.aux_exp_data
            except:
                raise Exception('failed to load aux_exp_data!!!, Please use stable_mode instantiation')

    def get_dataset_allsnr(self, dataset_type='train'):
        dataset = self.choose_dataset_type(dataset_type)
        X_train = dataset[0]
        labels = dataset[1]
        labels['target_type'] = [0 if y == 'animal' else 1 for y in labels['target_type']]
        return X_train, labels

    def get_dataset_test_allsnr(self):
        dataset = self.choose_dataset_type('test')
        X_train = dataset[0]
        return X_train

    def plot_single_timestamp_frequency(self, SAMPLE_NUMBER=0, timestep=0):
        '''
        Plotting the accuracy over different sweeps back to back
        '''
        path = os.path.join(os.getcwd(), 'graphs')
        if not os.path.isdir(path):
            os.mkdir(path)
        plt.figure()
        plt.plot(range(self.train_data[0].shape[1]), self.train_data[0][SAMPLE_NUMBER][:, timestep], linewidth=2,label='timestamp 0')
        plt.title('Single Timestamp Frequency', fontsize=30)
        plt.legend(loc="best")
        plt.xlabel('K[Bin index]', fontsize=20)
        plt.ylabel('Power', fontsize=30)
        fig_path = os.path.join(path, 'Single Timestamp Frequency')
        plt.savefig(fig_path)

    def plot_IQ_data(self,SAMPLE_NUMBER=0):
        path = os.path.join(os.getcwd(), 'graphs')
        if not os.path.isdir(path):
            os.mkdir(path)
        plt.figure()
        plt.imshow(self.train_data[0][SAMPLE_NUMBER])
        plt.title('IQ Matrix', fontsize=30)

        plt.legend(loc="best")
        plt.xlabel('Time step', fontsize=20)
        plt.ylabel('K[Bin index]', fontsize=30)
        fig_path = os.path.join(path, 'Single IQ Matrix')
        plt.savefig(fig_path)
