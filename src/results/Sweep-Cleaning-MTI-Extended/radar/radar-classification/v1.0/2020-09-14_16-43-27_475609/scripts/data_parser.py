import pickle
import os
import csv
import collections
import numpy as np
from os.path import expanduser
from matplotlib import pyplot as plt
import pandas as pd
from scipy import interpolate
from sklearn.decomposition import PCA
from data.signal_processing import *
import matplotlib


class DataSetParser(object):
    def __init__(self, stable_mode, config, read_test_only=False):
        self.radar_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
        self.dataset_dir = '{}/dataset'.format(self.radar_dir)
        self.dataset_pickle = os.listdir(self.dataset_dir)
        self.aux_exp_file_path = "MAFAT RADAR Challenge - Auxiliary Experiment Set V2"
        self.config = config
        self.is_test = read_test_only

        if read_test_only is True:
            self.test_data = self.dump_data(pickle_name='MAFAT RADAR Challenge - Public Test Set V1.pkl',
                                            csv_name='MAFAT RADAR Challenge - Public Test Set V1.csv')
            return

        # load train
        self.train_data = self.dump_data(pickle_name='MAFAT RADAR Challenge - Training Set V1.pkl',
                                         csv_name='MAFAT RADAR Challenge - Training Set V1.csv')

        if self.config.with_background_data is True:
            self.aux_background_data = self.dump_data(
                pickle_name='MAFAT RADAR Challenge - Auxiliary Background(empty) Set V2.pkl',
                csv_name='MAFAT RADAR Challenge - Auxiliary Background(empty) Set V2.csv')
            self.plot_histogram()

        self.aux_exp_data = None  # for future use from code
        if stable_mode:
            # self.aux_syn_data = self.dump_data(pickle_name='MAFAT RADAR Challenge - Auxiliary Synthetic Set V2.pkl',
            #                                    csv_name='MAFAT RADAR Challenge - Auxiliary Synthetic Set V2.csv')
            # self.aux_exp_data = self.dump_data(pickle_name='MAFAT RADAR Challenge - Auxiliary Experiment Set V2.pkl',
            #                                    csv_name='MAFAT RADAR Challenge - Auxiliary Experiment Set V2.csv')
            self.aux_exp_data = self.load_data_aux_exp_data()

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
        return np.fft.fftshift(iq)

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

    def aux_split(self, num_of_animals, num_of_humans):
        """
        Selects segments from the auxilary set for training set, in order to create a balanced Animal and human train set

        Arguments:
          num_of_animals, num_of_humans
        Returns:
          The auxilary data for the training, thge rate of exapnsion is: r = (N_A - N_H) / N_uniquie_tracks
        """
        data = self.aux_exp_data
        uniqe_tracks = np.unique(data['track_id'])
        N_unique = len(uniqe_tracks)
        r = round((num_of_animals - num_of_humans) / float(N_unique))
        idx = np.bool_(np.zeros(len(data['track_id'])))
        for track in uniqe_tracks:
            track_ind_total = np.where(data['track_id'] == track)[0]
            if len(track_ind_total) < r:
                track_indices = np.random.choice(track_ind_total, size=len(track_ind_total), replace=True).tolist()
            else:
                track_indices = np.random.choice(track_ind_total, size=r, replace=True).tolist()
            idx[track_indices] = True
            # idx |= data['segment_id'] == (data['segment_id'][data['track_id'] == track][:segments_per_aux_track])

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
    def mti_improvement(self, iq_mat):
        improved_iq = np.zeros((iq_mat.shape[0]-1,iq_mat.shape[1]),dtype=np.complex128)
        improved_iq[0,:] = iq_mat[0,:]
        for fast_time_sample_index in range(1,iq_mat.shape[0] - 1):
            improved_iq[fast_time_sample_index,:] = np.subtract(iq_mat[fast_time_sample_index,:], iq_mat[fast_time_sample_index - 1,:])
        return improved_iq

    def preprocess_pkl_data(self, pkl_raw_data):
        raw_iq_matrices = pkl_raw_data['iq_sweep_burst']
        doppler_burst = pkl_raw_data['doppler_burst']

        if self.config.use_mti_improvement is True:
            iq = np.zeros((raw_iq_matrices.shape[0], raw_iq_matrices.shape[1] - 3, raw_iq_matrices.shape[2]))
            for segment in range(raw_iq_matrices.shape[0]):
                iq[segment] = self.fft(self.mti_improvement(raw_iq_matrices[segment]))
                if self.config.with_max_value_on_doppler is True:
                    iq[segment] = self.max_value_on_doppler(iq=iq[segment], doppler_burst=doppler_burst[segment])
                else:
                    iq[segment] = np.fft.fftshift(iq[segment])
                iq[segment] = self.normalize(iq=iq[segment])
                if self.config.with_spectrum_cleaning is True:
                    iq[segment] = eClean_algorithm(iq[segment])
                elif self.config.with_kalman_filter_noise_estimation is True:
                    iq[segment] = kalman_filter_clean_algorithm(iq[segment])
        else:
            iq = np.zeros((raw_iq_matrices.shape[0], raw_iq_matrices.shape[1] - 2, raw_iq_matrices.shape[2]))
            for segment in range(raw_iq_matrices.shape[0]):
                self.PCA_expansion(raw_iq_matrices[segment])
                iq[segment] = self.fft(raw_iq_matrices[segment])
                if self.config.with_max_value_on_doppler is True:
                    iq[segment] = self.max_value_on_doppler(iq=iq[segment], doppler_burst=doppler_burst[segment])
                else:
                    iq[segment] = np.fft.fftshift(iq[segment])
                iq[segment] = self.normalize(iq=iq[segment])
                if self.config.with_spectrum_cleaning is True:
                    iq[segment] = eClean_algorithm(iq[segment])
                elif self.config.with_kalman_filter_noise_estimation is True:
                    iq[segment] = kalman_filter_clean_algorithm(iq[segment])
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
        plt.plot(range(self.train_data[0].shape[1]), self.train_data[0][SAMPLE_NUMBER][:, timestep], linewidth=2,
                 label='timestamp 0')
        plt.title('Single Timestamp Frequency', fontsize=30)
        plt.legend(loc="best")
        plt.xlabel('K[Bin index]', fontsize=20)
        plt.ylabel('Power', fontsize=30)
        fig_path = os.path.join(path, 'Single Timestamp Frequency')
        plt.savefig(fig_path)

    def plot_IQ_data(self, SAMPLE_NUMBER=0):
        path = os.path.join(os.getcwd(), 'graphs')
        if not os.path.isdir(path):
            os.mkdir(path)
        plt.figure()
        plt.imshow(self.train_data[0][SAMPLE_NUMBER])
        plt.title('Freq Response Matrix', fontsize=30)

        plt.legend(loc="best")
        plt.xlabel('Time step', fontsize=20)
        plt.ylabel('K[Bin index]', fontsize=30)
        fig_path = os.path.join(path, 'Single IQ Matrix')
        plt.savefig(fig_path)

    def plot_histogram(self):

        histogram, bin_edges = np.histogram(self.aux_background_data[0][0], bins=32 * 126, range=(np.min(self.aux_background_data[0][0]), np.max(self.aux_background_data[0][0])))
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Background')
        ax1.plot(np.linspace(np.min(self.aux_background_data[0][0]), np.max(self.aux_background_data[0][0]), 32 * 126),
                 histogram)
        ax1.set(xlabel='Value', ylabel='Count')
        ax1.set_title('Freq Response Histogram')

        ax2.imshow(self.aux_background_data[0][0])
        ax2.set(xlabel='Slow Axis', ylabel='Fast Axis')
        ax2.set_title('Freq Response Spectrogram')

        fig.savefig('Background')

        histogram, bin_edges = np.histogram(self.train_data[0][0], bins=32 * 126,
                                            range=(np.min(self.train_data[0][0]), np.max(self.train_data[0][0])))

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Human')
        ax1.plot(np.linspace(np.min(self.train_data[0][6000]), np.max(self.train_data[0][6000]), 32 * 126),
                 histogram)
        ax1.set(xlabel='Value', ylabel='Count')
        ax1.set_title('Freq Response Histogram')

        ax2.imshow(self.train_data[0][6000])
        ax2.set(xlabel='Slow Axis', ylabel='Fast Axis')
        ax2.set_title('Freq Response Spectrogram')

        fig.savefig('Human')

        histogram, bin_edges = np.histogram(self.train_data[0][0], bins=32 * 126,
                                            range=(np.min(self.train_data[0][0]), np.max(self.train_data[0][0])))

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Animal')
        ax1.plot(np.linspace(np.min(self.train_data[0][0]), np.max(self.train_data[0][0]), 32 * 126),
                 histogram)
        ax1.set(xlabel='Value', ylabel='Count')
        ax1.set_title('Freq Response Histogram')

        ax2.imshow(self.train_data[0][0])
        ax2.set(xlabel='Slow Axis', ylabel='Fast Axis')
        ax2.set_title('Freq Response Spectrogram')

        fig.savefig('Animal')

    def plot_interpolation_data(self):
        path = os.path.join(os.getcwd(), 'graphs')

        if not os.path.isdir(path):
            os.mkdir(path)

        plt.figure()
        plt.imshow(self.interpolated_data[0][0])
        plt.title('IQ Matrix Original', fontsize=30)
        plt.legend(loc="best")
        plt.xlabel('Time step', fontsize=20)
        plt.ylabel('K[Bin index]', fontsize=30)
        fig_path = os.path.join(path, 'Original IQ Matrix')
        plt.savefig(fig_path)

        plt.figure()
        plt.imshow(self.interpolated_data[0][1])
        plt.title('IQ Matrix Interpolated', fontsize=30)
        plt.legend(loc="best")
        plt.xlabel('Time step', fontsize=20)
        plt.ylabel('K[Bin index]', fontsize=30)
        fig_path = os.path.join(path, 'Interpolated IQ Matrix')
        plt.savefig(fig_path)

    def interpolation_fast_axis(self, iq_time_matrix):
        '''
        Creating the real axis
        '''

        iq_normalized = (np.real(iq_time_matrix) - np.mean(np.real(iq_time_matrix))) / np.std(np.real(iq_time_matrix)) \
                        + 1j * (np.imag(iq_time_matrix) - np.mean(np.imag(iq_time_matrix))) / np.std(np.imag(iq_time_matrix))

        slow_axis = np.arange(0, 32)
        fast_axis = np.arange(0, 128)
        mesh_fast_axis, mesh_slow_axis = np.meshgrid(slow_axis, fast_axis)

        '''
        Generating the interpolator object
        '''
        interpolated_iq_time_func_real = [interpolate.interp1d(fast_axis,np.real(iq_normalized[:,timestep_index]),kind='linear') for timestep_index in range(iq_normalized.shape[1])]
        interpolated_iq_time_func_imag = [interpolate.interp1d(fast_axis,np.imag(iq_normalized[:,timestep_index]),kind='linear') for timestep_index in range(iq_normalized.shape[1])]

        fast_axis_interpolated = np.arange(0, 128, 1. / self.config.interpolation_size)


        interpolated_iq_list = []
        for index in range(self.config.interpolation_size):
            start_index = index
            fast_stop_index = self.config.interpolation_size * 128

            # Generate the current fast axis indices
            current_fast_axis_indices = np.arange(start_index, fast_stop_index, self.config.interpolation_size)
            # sample the fast axis with the current indices
            sampled_fast_axis = np.take(fast_axis_interpolated, current_fast_axis_indices)

            if sampled_fast_axis[-1] > 127:
                '''
                Handle with the case of sampling outside
                '''
                sampled_fast_axis[-1] = 127

            interpolated_iq_mat_real = np.stack([item(sampled_fast_axis) for item in interpolated_iq_time_func_real])
            interpolated_iq_mat_imag = np.stack([item(sampled_fast_axis) for item in interpolated_iq_time_func_imag])
            interpolated_iq_list.append(interpolated_iq_mat_real
                                        + 1j * interpolated_iq_mat_imag)

        return interpolated_iq_list

    def interpolation_slow_axis(self, iq_time_matrix):
        '''
        Creating the real axis
        '''

        iq_normalized = (np.real(iq_time_matrix) - np.mean(np.real(iq_time_matrix))) / np.std(np.real(iq_time_matrix)) \
                        + 1j * (np.imag(iq_time_matrix) - np.mean(np.imag(iq_time_matrix))) / np.std(np.imag(iq_time_matrix))

        slow_axis = np.arange(0, 32)
        fast_axis = np.arange(0, 128)
        mesh_fast_axis, mesh_slow_axis = np.meshgrid(slow_axis, fast_axis)

        '''
        Generating the interpolator object
        '''
        interpolated_iq_time_func_real = [interpolate.interp1d(slow_axis,np.real(iq_normalized[freq_index,:]),kind='linear') for freq_index in range(iq_normalized.shape[0])]
        interpolated_iq_time_func_imag = [interpolate.interp1d(slow_axis,np.imag(iq_normalized[freq_index,:]),kind='linear') for freq_index in range(iq_normalized.shape[0])]

        slow_axis_interpolated = np.arange(0, 32, 1. / self.config.interpolation_size)


        interpolated_iq_list = []
        for index in range(self.config.interpolation_size):
            start_index = index
            slow_stop_index = self.config.interpolation_size * 32

            # Generate the current slow axis indices
            current_slow_axis_indices = np.arange(start_index, slow_stop_index, self.config.interpolation_size)
            # sample the slow axis with the current indices
            sampled_slow_axis = np.take(slow_axis_interpolated, current_slow_axis_indices)

            if sampled_slow_axis[-1] > 31:
                '''
                Handle with the case of sampling outside of the range
                '''
                sampled_slow_axis[-1] = 31

            interpolated_iq_mat_real = np.stack([item(sampled_slow_axis) for item in interpolated_iq_time_func_real])
            interpolated_iq_mat_imag = np.stack([item(sampled_slow_axis) for item in interpolated_iq_time_func_imag])
            interpolated_iq_list.append(np.transpose(interpolated_iq_mat_real
                                        + 1j * interpolated_iq_mat_imag))

        return interpolated_iq_list

    def generate_interpolated_iq_samples(self, pkl_raw_data,validation_indices,interpolation_func):
        doppler_burst = pkl_raw_data['doppler_burst']
        raw_iq_matrices = pkl_raw_data['iq_sweep_burst']
        interpolation_size = self.config.interpolation_size
        train_segments_id = np.arange(0, raw_iq_matrices.shape[0])
        train_segments_id = np.delete(train_segments_id, validation_indices)# remove the validation indices

        if self.config.use_mti_improvement is True:
            '''
            With MTI
            '''
            iq = np.zeros((interpolation_size * raw_iq_matrices.shape[0] - len(validation_indices), raw_iq_matrices.shape[1] - 3,
                           raw_iq_matrices.shape[2]))
            for segment, segment_id in zip(range(0, iq.shape[0], interpolation_size), train_segments_id):
                for interpolate_index, current_iq_mat in zip(range(interpolation_size), interpolation_func(raw_iq_matrices[segment_id])):
                    iq[segment + interpolate_index] = self.fft(self.mti_improvement(np.transpose(current_iq_mat)))
                    iq[segment + interpolate_index] = self.max_value_on_doppler(iq=iq[interpolate_index + segment],
                                                                                doppler_burst=doppler_burst[segment_id])
                    iq[segment + interpolate_index] = self.normalize(iq=iq[interpolate_index + segment])
                    if self.config.with_spectrum_cleaning is True:
                        iq[segment + interpolate_index] = eClean_algorithm(iq[segment + interpolate_index])
                    elif self.config.with_kalman_filter_noise_estimation is True:
                        iq[segment] = kalman_filter_clean_algorithm(iq[segment])
        else:
            '''
            Without MTI
            '''
            iq = np.zeros((interpolation_size * raw_iq_matrices.shape[0] - len(validation_indices), raw_iq_matrices.shape[1] - 2,
                           raw_iq_matrices.shape[2]))
            for segment, segment_id in zip(range(0, iq.shape[0], interpolation_size), train_segments_id):
                for interpolate_index, current_iq_mat in zip(range(interpolation_size), interpolation_func(raw_iq_matrices[segment_id])):
                    iq[segment + interpolate_index] = self.fft(np.transpose(current_iq_mat))
                    iq[segment + interpolate_index] = self.max_value_on_doppler(iq=iq[interpolate_index + segment],
                                                                                doppler_burst=doppler_burst[segment_id])
                    iq[segment + interpolate_index] = self.normalize(iq=iq[interpolate_index + segment])
                    if self.config.with_spectrum_cleaning is True:
                        iq[segment + interpolate_index] = eClean_algorithm(iq[segment + interpolate_index])
                    elif self.config.with_kalman_filter_noise_estimation is True:
                        iq[segment] = kalman_filter_clean_algorithm(iq[segment])
        return iq

    def generate_interpolated_csv_data(self, csv_labels,validation_indices):
        is_first_line = True
        labels_augmented = collections.OrderedDict()

        for line in csv_labels:
            if is_first_line:
                is_first_line = False
                for key in line:
                    labels_augmented[key] = []
            else:
                line_is_validation_data = False

                # Check if this line contains a data this is a part of the validation data set
                for key, data in zip(labels_augmented.keys(), line):
                    if key == 'segment_id':
                        if data in validation_indices:
                            line_is_validation_data = True
                        break

                if line_is_validation_data is False:
                    for key, data in zip(labels_augmented.keys(), line):
                        for index in range(self.config.interpolation_size):
                            # Duplicate the data
                            if key == 'segment_id':
                                labels_augmented[key].append(str(-int(data)))
                            else:
                                labels_augmented[key].append(data)

        return labels_augmented

    def generate_pca_csv_data(self, csv_labels,validation_indices):
        is_first_line = True
        labels_augmented = collections.OrderedDict()

        for line in csv_labels:
            if is_first_line:
                is_first_line = False
                for key in line:
                    labels_augmented[key] = []
            else:
                line_is_validation_data = False

                # Check if this line contains a data this is a part of the validation data set
                for key, data in zip(labels_augmented.keys(), line):
                    if key == 'segment_id':
                        if data in validation_indices:
                            line_is_validation_data = True
                        break

                if line_is_validation_data is False:
                    for key, data in zip(labels_augmented.keys(), line):
                        for index in range(self.config.num_of_pca_segments):
                            # Duplicate the data
                            if key == 'segment_id':
                                labels_augmented[key].append(str(-int(data)))
                            else:
                                labels_augmented[key].append(data)

        return labels_augmented

    def generate_pca_iq_samples(self,pkl_raw_data, validation_indices):
        doppler_burst = pkl_raw_data['doppler_burst']
        raw_iq_matrices = pkl_raw_data['iq_sweep_burst']
        pca_size = self.config.num_of_pca_segments
        train_segments_id = np.arange(0, raw_iq_matrices.shape[0])
        train_segments_id = np.delete(train_segments_id, validation_indices)  # remove the validation indices

        if self.config.use_mti_improvement is True:
            '''
            With MTI
            '''
            iq = np.zeros((pca_size * raw_iq_matrices.shape[0] - len(validation_indices), raw_iq_matrices.shape[1] - 3,
                           raw_iq_matrices.shape[2]))
            for segment, segment_id in zip(range(0, iq.shape[0], pca_size), train_segments_id):
                for pca_index, current_iq_mat in zip(range(pca_size), self.PCA_expansion(raw_iq_matrices[segment_id])):
                    iq[segment + pca_index] = self.fft(self.mti_improvement(current_iq_mat))
                    if self.config.with_max_value_on_doppler is True:
                        iq[segment + pca_index] = self.max_value_on_doppler(iq=iq[segment + pca_index],
                                                                                    doppler_burst=doppler_burst[segment_id])
                    else:
                        iq[segment + pca_index] = np.fft.fftshift(iq[segment + pca_index])
                    iq[segment + pca_index] = self.normalize(iq=iq[pca_index + segment])
                    if self.config.with_spectrum_cleaning is True:
                        iq[segment + pca_index] = eClean_algorithm(iq[segment + pca_index])
                    elif self.config.with_kalman_filter_noise_estimation is True:
                        iq[segment + pca_index] = kalman_filter_clean_algorithm(iq[segment + pca_index])
        else:
            '''
            Without MTI
            '''
            iq = np.zeros((pca_size * raw_iq_matrices.shape[0] - len(validation_indices), raw_iq_matrices.shape[1] - 2,
                           raw_iq_matrices.shape[2]))
            for segment, segment_id in zip(range(0, iq.shape[0], pca_size), train_segments_id):
                for pca_index, current_iq_mat in zip(range(pca_size), self.PCA_expansion(raw_iq_matrices[segment_id])):
                    iq[segment + pca_index] = self.fft(current_iq_mat)
                    if self.config.with_max_value_on_doppler is True:
                        iq[segment + pca_index] = self.max_value_on_doppler(iq=iq[pca_index + segment],
                                                                                    doppler_burst=doppler_burst[segment_id])
                    else:
                        iq[segment + pca_index] = np.fft.fftshift(iq[segment + pca_index])
                    if self.config.with_spectrum_cleaning is True:
                        iq[segment + pca_index] = eClean_algorithm(iq[segment + pca_index])
                    elif self.config.with_kalman_filter_noise_estimation is True:
                        iq[segment + pca_index] = kalman_filter_clean_algorithm(iq[segment + pca_index])
        return iq

    def generate_pca_expansion(self,pickle_name,csv_name,validation_indices):
        os.chdir(self.dataset_dir)
        with open('{}'.format(pickle_name), 'rb') as pickle_file:
            raw_data = pickle.load(pickle_file)
            iq = self.generate_pca_iq_samples(pkl_raw_data=raw_data, validation_indices=validation_indices)

        with open('{}'.format(csv_name)) as csv_file:
            labels = csv.reader(csv_file, delimiter=',', quotechar='|')
            labels_dict = self.generate_pca_csv_data(csv_labels=labels, validation_indices=validation_indices)

        return iq, labels_dict

    def PCA_expansion(self, iq_mat):
        '''
        Calculate the PCA of the vector and generate samples from it
        '''
        standardized_iq_mat = (iq_mat-np.mean(iq_mat,axis=0))
        cov_mat = np.cov(standardized_iq_mat)

        '''
        Eigen Value decomposition
        '''
        eigen_values, eigen_vector = np.linalg.eigh(cov_mat)  # Calculate the covariance matrix
        idx = np.argsort(eigen_values)[::-1]
        eigen_vector = eigen_vector[:, idx]
        '''
        Generate new samples
        '''
        NUMBER_OF_KL_COEEFIECENT = self.config.number_of_pca_coeff
        VARIANCE_SCALING = self.config.pca_augmentation_scaling

        new_iq_matrices = []
        for expansion_index in range(self.config.num_of_pca_segments):
            KL_mat = np.matmul(np.transpose(np.conjugate(standardized_iq_mat)), eigen_vector)
            KL_mat[:, :NUMBER_OF_KL_COEEFIECENT] = (np.real(KL_mat[:,:NUMBER_OF_KL_COEEFIECENT]) + np.sqrt(VARIANCE_SCALING) * np.multiply(np.real(KL_mat[:,:NUMBER_OF_KL_COEEFIECENT]),((1/(np.sqrt(2)))*np.random.randn(32, NUMBER_OF_KL_COEEFIECENT)))) + \
                                                   (1.0j * ( np.imag(KL_mat[:,:NUMBER_OF_KL_COEEFIECENT]) + np.sqrt(VARIANCE_SCALING) * np.multiply(np.imag(KL_mat[:,:NUMBER_OF_KL_COEEFIECENT]),((1/(np.sqrt(2)))*np.random.randn(32, NUMBER_OF_KL_COEEFIECENT)))))
            KL_mat[:, :] = KL_mat[:, :] * np.exp(2.0j*np.pi*np.random.randn(1) * 0.1)
            new_iq_mat = np.matmul(eigen_vector,np.transpose(np.conjugate(KL_mat)))
            new_iq_matrices.append(new_iq_mat)

        # figure, axes = plt.subplots(nrows=2, ncols=2)
        # axes[0,0].imshow(self.fft(standardized_iq_mat))
        # axes[0,0].set_title('Freq Response Org')
        # axes[0,1].imshow(self.fft(new_iq_matrices[0]))
        # axes[0,1].set_title('Freq Response Generated 1 ')
        # axes[1,0].imshow(self.fft(new_iq_matrices[1]))
        # axes[1,0].set_title('Freq Response Generated 2 ')
        # axes[1,1].imshow(self.fft(new_iq_matrices[2]))
        # axes[1,1].set_title('Freq Response Generated 3 ')
        #
        # figure.savefig('test_pca')
        return new_iq_matrices

    def get_pca_synthetic_data(self,validation_segments_id_list):
        self.PCA_synthetic_data = self.generate_pca_expansion(pickle_name='MAFAT RADAR Challenge - Training Set V1.pkl',
                                                            csv_name='MAFAT RADAR Challenge - Training Set V1.csv',
                                                            validation_indices=validation_segments_id_list)
        self.PCA_synthetic_data[1]['target_type'] = [0 if y == 'animal' else 1 for y in self.PCA_synthetic_data[1]['target_type']]
        return self.PCA_synthetic_data

    def dump_interpolated_data(self, pickle_name, csv_name,validation_indices,interpolation_func):
        os.chdir(self.dataset_dir)
        with open('{}'.format(pickle_name), 'rb') as pickle_file:
            raw_data = pickle.load(pickle_file)
            iq = self.generate_interpolated_iq_samples(pkl_raw_data=raw_data,validation_indices=validation_indices,interpolation_func=interpolation_func)

        with open('{}'.format(csv_name)) as csv_file:
            labels = csv.reader(csv_file, delimiter=',', quotechar='|')
            labels_dict = self.generate_interpolated_csv_data(csv_labels=labels,validation_indices=validation_indices)

        return iq, labels_dict

    def get_interpolation_data_fast_axis(self,validation_segments_id_list):
        self.interpolated_data = self.dump_interpolated_data(pickle_name='MAFAT RADAR Challenge - Training Set V1.pkl',
                                                            csv_name='MAFAT RADAR Challenge - Training Set V1.csv',
                                                            validation_indices=validation_segments_id_list,
                                                            interpolation_func=self.interpolation_fast_axis)
        self.plot_interpolation_data()
        self.interpolated_data[1]['target_type'] = [0 if y == 'animal' else 1 for y in self.interpolated_data[1]['target_type']]
        return self.interpolated_data

    def get_interpolation_data_slow_axis(self,validation_segments_id_list):
        self.interpolated_data = self.dump_interpolated_data(pickle_name='MAFAT RADAR Challenge - Training Set V1.pkl',
                                                            csv_name='MAFAT RADAR Challenge - Training Set V1.csv',
                                                            validation_indices=validation_segments_id_list,
                                                            interpolation_func=self.interpolation_slow_axis)
        self.plot_interpolation_data()
        self.interpolated_data[1]['target_type'] = [0 if y == 'animal' else 1 for y in self.interpolated_data[1]['target_type']]
        return self.interpolated_data

