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
import scipy.signal
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

        if self.config.with_background_data or self.config.learn_background:
            self.aux_background_data = self.dump_data(
                pickle_name='MAFAT RADAR Challenge - Auxiliary Background Set V2.pkl',
                csv_name='MAFAT RADAR Challenge - Auxiliary Background Set V2.csv')
            # self.plot_histogram()

        self.aux_exp_data = None  # for future use from code

        if stable_mode:
            self.aux_exp_data = self.load_data_aux_exp_data()
        if self.config.load_low_human_experiment:
            self.aux_exp_data = self.dump_data(
                pickle_name='MAFAT RADAR Challenge - Auxiliary Experiment Set V2.pkl',
                csv_name='MAFAT RADAR Challenge - Auxiliary Experiment Set V2.csv')
        # self.plot_histogram()
        if self.config.load_synthetic:
            # 'MAFAT RADAR Challenge - Auxiliary Synthetic Set V2'
            self.aux_syn = self.dump_data(
                pickle_name='MAFAT RADAR Challenge - Auxiliary Synthetic Set V2.pkl',
                csv_name='MAFAT RADAR Challenge - Auxiliary Synthetic Set V2.csv')

    def fft(self, iq, window_type='hann',axis=0, window=None):
        """
          Computes the log of discrete Fourier Transform (DFT).

          Arguments:
            iq_burst -- {ndarray} -- 'iq_sweep_burst' array
            axis -- {int} -- axis to perform fft in (Default = 0)

          Returns:
            log of DFT on iq_burst array
        """
        window_option = {'hann': self.hann, 'blackman': self.blackman, 'blackman_nutall': self.blackman_nutall}

        assert window_type == 'hann' or window_type == 'blackman' or window_type == 'blackman_nutall'

        smooth_iq = window_option[window_type](iq, window)

        assert self.config.with_iq_matrices != True or self.config.with_magnitude_phase != True

        if self.config.with_iq_matrices is True:
            iq = np.zeros(shape=(smooth_iq.shape[0],smooth_iq.shape[1],2))
            both_iq = np.fft.fft(smooth_iq, axis=axis)
            iq[:, :, 0] = np.real(both_iq)
            iq[:, :, 1] = np.imag(both_iq)
        elif self.config.with_magnitude_phase is True:
            iq = np.zeros(shape=(smooth_iq.shape[0], smooth_iq.shape[1], 2))
            both_iq = np.fft.fft(smooth_iq, axis=axis)
            iq[:, :, 0] = np.log(np.abs(both_iq))
            iq[:, :, 1] = np.unwrap(np.angle(both_iq))
        else:
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
        if self.config.with_magnitude_phase is True:
            iq[:,:,0] = np.maximum(np.median(iq[:,:,0]) - 1, iq[:,:,0])
        if self.config.with_iq_matrices is False:
            iq = np.maximum(np.median(iq)-1, iq)
            iq = (iq - np.mean(iq)) / np.std(iq)
        else:
            iq = iq / np.max(np.abs(iq[:,:,0]+1j*iq[:,:,1]))
        return iq

    '''
    Windowing functions
    '''
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

    def blackman(self, iq, window=None):
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
        alpha = self.config.blackman_alpha
        alpha_0 = 0.5 * (1 - alpha)
        alpha_1 = 0.5
        alpha_2 = alpha / 2

        blackmanCol = alpha_0 - alpha_1 * np.cos(2 * np.pi * (n / N)) + alpha_2 * np.cos(4 * np.pi * (n / N))
        return (blackmanCol * iq[window[0]:window[1]])[1:-1]

    def blackman_nutall(self, iq, window=None):
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
        alpha = self.config.blackman_alpha
        alpha_0 = 0.3635819
        alpha_1 = 0.4891775
        alpha_2 = 0.1365995
        alpha_3 = 0.0106411

        blackmanCol = alpha_0 - alpha_1 * np.cos(2 * np.pi * (n / N)) + alpha_2 * np.cos(4 * np.pi * (n / N)) - alpha_3 * np.cos(6 * np.pi * (n / N))
        return (blackmanCol * iq[window[0]:window[1]])[1:-1]

    def aux_split(self, num_of_animals, num_of_humans):
        """
        Selects segments from the auxilary set for training set, in order to create a balanced Animal and human train set

        Arguments:
          num_of_animals, num_of_humans
        Returns:
          The auxilary data for the training, the rate of exapnsion is: r = (N_A - N_H) / N_unique_tracks
        """
        if self.config.snr_type == 'all':
            data = self.aux_exp_data
        elif self.config.snr_type == 'low':
            data = self.get_dataset_by_snr(dataset_type='aux_exp', snr_type='LowSNR')
        elif self.config.snr_type == 'high':
            data = self.get_dataset_by_snr(dataset_type='aux_exp', snr_type='HighSNR')
        uniqe_tracks = np.unique(data['track_id'])
        N_unique = len(uniqe_tracks)
        r = round((num_of_animals - num_of_humans) / float(N_unique))
        idx = np.bool_(np.zeros(len(data['track_id'])))
        for track in uniqe_tracks:
            track_ind_total = np.where(data['track_id'] == track)[0]
            track_indices = np.random.choice(track_ind_total, size=min(len(track_ind_total), r), replace=False).tolist()
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

    def load_all_experiment(self):
        if self.config.snr_type == 'all':
            data = self.aux_exp_data
        elif self.config.snr_type == 'low':
            data = self.get_dataset_by_snr(dataset_type='aux_exp', snr_type='LowSNR')
        elif self.config.snr_type == 'high':
            data = self.get_dataset_by_snr(dataset_type='aux_exp', snr_type='HighSNR')
        else:
            raise Exception('data not initallized')

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

        bytes_in = bytearray(0)
        input_size = os.path.getsize('{}.pkl'.format(file_path))
        max_bytes = 2 ** 31 - 1

        with open('{}.pkl'.format(file_path), 'rb') as data:
            for _ in range(0, input_size, max_bytes):
                bytes_in += data.read(max_bytes)
        output = pickle.loads(bytes_in)
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

    def preprocess_pkl_data(self, pkl_raw_data):
        raw_iq_matrices = pkl_raw_data['iq_sweep_burst']
        doppler_burst = pkl_raw_data['doppler_burst']

        if self.config.with_iq_matrices is True or self.config.with_magnitude_phase is True:
            iq = np.zeros((raw_iq_matrices.shape[0], raw_iq_matrices.shape[1] - 2, raw_iq_matrices.shape[2], 2))
        else:
            iq = np.zeros((raw_iq_matrices.shape[0], raw_iq_matrices.shape[1] - 2, raw_iq_matrices.shape[2]))
        for segment in range(raw_iq_matrices.shape[0]):
            iq[segment] = self.fft(raw_iq_matrices[segment], window_type=self.config.default_window_type)
            if self.config.with_max_value_on_doppler is True:
                iq[segment] = self.max_value_on_doppler(iq=iq[segment], doppler_burst=doppler_burst[segment])
            else:
                iq[segment] = np.fft.fftshift(iq[segment])
            iq[segment] = self.normalize(iq=iq[segment])

        # figure, axes = plt.subplots(nrows=1, ncols=2)
        # axes[0].imshow(iq[segment][:, :, 0])
        # axes[0].set_title('I')
        # axes[1].imshow(iq[segment][:, :, 1])
        # axes[1].set_title('Q')
        # figure.savefig('I_Q_Freq')
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
        # if self.config.snr_type == 'all':
        #     X_train = dataset[0]
        #     labels = dataset[1]
        #     labels['target_type'] = [0 if y == 'animal' else 1 for y in labels['target_type']]
        #     return X_train, labels

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
        elif dataset_type is 'aux_syn':
            try:
                return self.aux_syn
            except:
                raise Exception('failed to load aux_syn!!!, Please use stable_mode instantiation')

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

    def get_dataset_by_snr(self,dataset_type,snr_type):
        '''
        snr_type can be HighSNR or LowSNR
        '''
        if snr_type != 'HighSNR' and snr_type != 'LowSNR':
            raise Exception('Unvalid SNR type please enter a valid one')

        dataset = self.choose_dataset_type(dataset_type)

        if 'aux_exp' == dataset_type:
            return dataset
        else:
            X = dataset[0]
            labels = dataset[1]
            new_X = []
            new_labels = dict.fromkeys(labels.keys()) # copy the keys
            for key in new_labels.keys():
                new_labels[key] = []

            for index,SNR in zip(range(X.shape[0]),labels['snr_type']):
                if SNR == snr_type:
                    for key in new_labels.keys():
                        new_labels[key].append(labels[key][index])
                    new_X.append(X[index])

            new_labels['target_type'] = [0 if y == 'animal' else 1 for y in new_labels['target_type']]

            return np.array(new_X), new_labels

    '''
    Visulaization
    '''
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
        mean_histogram = np.zeros((self.aux_background_data[0][0].shape[0],1))
        MAX_VALUES_THRESHOLD = 10
        for segment in range(len(self.aux_background_data[0])):
            for time_step_index in range(self.aux_background_data[0][0].shape[1]):
                indices = (-self.aux_background_data[0][segment][:,time_step_index]).argsort()[:MAX_VALUES_THRESHOLD]
                mean_histogram[indices] += 1
        mean_histogram /= len(self.aux_background_data[0])


        plt.figure()
        plt.plot(np.linspace(0,126,126),mean_histogram)
        plt.xlabel('Frequency Bin')
        plt.ylabel('Count')
        plt.title('Background Spectrogram Histogram')
        plt.savefig('Background Averaged')

        mean_histogram_human = np.zeros((self.train_data[0][0].shape[0],1))
        mean_histogram_animal = np.zeros((self.train_data[0][0].shape[0],1))
        animal_counter = 0
        human_counter = 0

        for segment in range(len(self.train_data[0])):
            for time_step_index in range(self.train_data[0][0].shape[1]):
                indices = (-self.train_data[0][segment][:,time_step_index]).argsort()[:MAX_VALUES_THRESHOLD]
                if self.train_data[1]['target_type'][segment] == 'animal':
                    mean_histogram_animal[indices] += 1
                else:
                    mean_histogram_human[indices] += 1
            if self.train_data[1]['target_type'][segment] == 'animal':
                animal_counter += 1
            else:
                human_counter += 1

        mean_histogram_animal /= animal_counter
        mean_histogram_human /= human_counter

        plt.figure()

        plt.plot(np.linspace(0, 126, 126),mean_histogram_human)
        plt.xlabel('Frequency Bin')
        plt.ylabel('Count')
        plt.title('Human Spectrogram Histogram')
        plt.savefig('Human Averaged Histograms')



        # ax2.imshow(self.train_data[0][6000])
        # ax2.set(xlabel='Slow Axis', ylabel='Fast Axis')
        # ax2.set_title('Freq Response Spectrogram')
        # fig.savefig('Human')
        #
        # mean_histogram = np.zeros((self.train_data[0][0].shape[0],1))
        # for time_step_index in range(self.train_data[0][0].shape[1]):
        #     indices = (-self.train_data[0][0][:, time_step_index]).argsort()[:MAX_VALUES_THRESHOLD]
        #     mean_histogram[indices] += 1
        #
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # fig.suptitle('Animal')
        plt.figure()
        plt.plot(np.linspace(0, 126, 126), mean_histogram_animal)
        plt.xlabel('Frequency Bin')
        plt.ylabel('Count')
        plt.title('Animal Spectrogram Histogram')
        plt.savefig('Animal Averaged Histograms')

        # ax2.imshow(self.train_data[0][0])
        # ax2.set(xlabel='Slow Axis', ylabel='Fast Axis')
        # ax2.set_title('Freq Response Spectrogram')

    def plot_different_window_functions(self,time_iq):
        temp_iq = np.copy(time_iq)
        smooth_iq = self.hann(time_iq)
        iq = np.log(np.abs(np.fft.fft(smooth_iq, axis=0)))
        plt.figure()
        plt.imshow(self.normalize(iq))
        plt.title('Hann')
        plt.savefig('Hann_Frequency_example')

        smooth_iq = self.blackman(temp_iq)
        iq_blackman = np.log(np.abs(np.fft.fft(smooth_iq, axis=0)))
        plt.figure()
        plt.imshow(self.normalize(iq_blackman))
        plt.title('Blackman')
        plt.savefig('Blackman_Frequency_example')

        smooth_iq = self.blackman_nutall(temp_iq)
        iq_nutall = np.log(np.abs(np.fft.fft(smooth_iq, axis=0)))
        plt.figure()
        plt.imshow(self.normalize(iq_nutall))
        plt.title('Blackman Nutall')
        plt.savefig('Blackman_nutall_Frequency_example')

        print('Blackman covriance distance {}'.format(np.linalg.norm(np.cov(iq_blackman-iq))/np.linalg.norm(np.cov(iq))))
        print('Blackman crosscorrelation distance {}'.format(np.linalg.norm(scipy.signal.correlate2d(iq_blackman,iq))))

        print('Nutall covriance distance {}'.format(np.linalg.norm(np.cov(iq_nutall-iq))/np.linalg.norm(np.cov(iq))))
        print('Nutall crosscorrelation distance {}'.format(np.linalg.norm(scipy.signal.correlate2d(iq_nutall,iq))))

    '''
    PCA Augmentation
    '''
    def analyse_PCA_hyperparameters(self,iq_list):
        var_list = np.linspace(start=0.05, stop=1.0, num=20)
        cov_array = np.zeros(shape=(len(iq_list),len(var_list)))
        cross_correlation_array = np.zeros(shape=(len(iq_list),len(var_list)))
        for iq,segment_index in zip(iq_list,range(len(iq_list))):
            standardized_iq_mat = iq - np.mean(iq, axis=0)
            for variance,var_ind in zip(var_list,range(len(var_list))):
                self.config.__setattr__("pca_augmentation_scaling", variance)
                X_augment = self.PCA_time_augmentation(iq)
                cov_array[segment_index][var_ind] = np.linalg.norm(np.cov(standardized_iq_mat) - np.cov(X_augment[0])) / np.linalg.norm(np.cov(standardized_iq_mat))
                cross_correlation_array[segment_index][var_ind] = scipy.signal.correlate2d(standardized_iq_mat,X_augment[0]) / scipy.signal.correlate2d(standardized_iq_mat,standardized_iq_mat)
        plt.figure()
        plt.plot(np.linspace(start=0.05, stop=1.0, num=20),np.mean(cov_array,axis=1))
        plt.title('Normalized PCA Covariance')
        plt.xlabel('Variance')
        plt.savefig("Normalized PCA Covariance to Variance")

        plt.figure()
        plt.plot(np.linspace(start=0.05, stop=1.0, num=20), np.mean(cross_correlation_array, axis=1))
        plt.title('Normalized PCA Cross Correlation')
        plt.xlabel('Variance')
        plt.savefig("Normalized PCA Cross Correlation")

    def PCA_time_augmentation(self, iq):
        '''
        Calculate the PCA of the vector and generate samples from it
        '''

        standardized_iq_mat = iq - np.mean(iq, axis=0)
        cov_mat = np.cov(standardized_iq_mat)  # Calculate the covariance matrix

        '''
        Eigen Value decomposition
        '''
        eigen_values, eigen_vector = np.linalg.eigh(cov_mat)  # Calculate the eigen values and eigen vectors
        idx = np.argsort(eigen_values)[::-1]
        eigen_vector = eigen_vector[:, idx]

        '''
        Generate new samples
        '''
        NUMBER_OF_KL_COEEFIECENT = self.config.number_of_pca_coeff
        VARIANCE_SCALING = self.config.pca_augmentation_scaling
        X_augment = []
        for test_index in range(self.config.num_of_pca_segments):
            KL_mat = np.matmul(np.transpose(np.conjugate(standardized_iq_mat)), eigen_vector)
            KL_mat[:, :NUMBER_OF_KL_COEEFIECENT] = KL_mat[:, :NUMBER_OF_KL_COEEFIECENT] + VARIANCE_SCALING * (np.multiply(KL_mat[:, :NUMBER_OF_KL_COEEFIECENT], np.random.normal(loc=0, scale=np.sqrt(2) / 2, size=(32, 2 * NUMBER_OF_KL_COEEFIECENT)).view(np.complex128)))
            new_iq_mat = np.matmul(eigen_vector, np.transpose(np.conjugate(KL_mat)))
            X_augment.append(new_iq_mat)

        # figure, axes = plt.subplots(nrows=2, ncols=2)
        # axes[0, 0].imshow(self.normalize(self.fft(standardized_iq_mat)))
        # axes[0, 0].set_title('Freq Response Org')
        # axes[0, 1].imshow(self.normalize(self.fft(X_augment[0])))
        # axes[0, 1].set_title('Freq Response Generated 1 ')
        # axes[1, 0].imshow(self.normalize(self.fft(X_augment[1])))
        # axes[1, 0].set_title('Freq Response Generated 2 ')
        # axes[1, 1].imshow(self.normalize(self.fft(X_augment[2])))
        # axes[1, 1].set_title('Freq Response Generated 3 ')
        # figure.savefig('test_pca')
        #
        # plt.figure()
        # plt.imshow(self.normalize(self.fft(standardized_iq_mat)))
        # plt.savefig('Org IQ')
        #
        # for x,index in zip(X_augment,range(len(X_augment))):
        #     plt.figure()
        #     plt.imshow(self.normalize(self.fft(x)))
        #     print('error for index {} is {}'.format(index,np.linalg.norm(np.cov(standardized_iq_mat) - np.cov(X_augment[index])) / np.linalg.norm(np.cov(standardized_iq_mat))))
        #     plt.savefig('Augmentated IQ{}'.format(index))

        return X_augment

    def generate_pca_expansion_for_iq_samples(self,pkl_raw_data, validation_indices):
        doppler_burst = pkl_raw_data['doppler_burst']
        raw_iq_matrices = pkl_raw_data['iq_sweep_burst']
        raw_iq_segments = pkl_raw_data['segment_id']
        expansion_size = self.config.num_of_pca_segments
        train_segments_id = [segment_id for segment_id in raw_iq_segments.tolist() if str(segment_id) not in validation_indices]  # remove the validation indices

        if self.config.with_iq_matrices is True or self.config.with_magnitude_phase is True:
            iq = np.zeros((expansion_size * len(train_segments_id), raw_iq_matrices.shape[1] - 2, raw_iq_matrices.shape[2], 2))
        else:
            iq = np.zeros((expansion_size * len(train_segments_id), raw_iq_matrices.shape[1] - 2, raw_iq_matrices.shape[2]))
        error_list = []
        for segment, segment_id in enumerate(train_segments_id):
            idx = train_segments_id.index(segment_id)
            for pca_index, current_iq_mat in zip(range(expansion_size), self.PCA_time_augmentation(raw_iq_matrices[idx])):

                current_index = self.config.num_of_pca_segments * segment + pca_index
                iq[current_index] = self.fft(current_iq_mat)
                if self.config.with_max_value_on_doppler is True:
                    iq[current_index] = self.max_value_on_doppler(iq=iq[current_index], doppler_burst=doppler_burst[idx])
                else:
                    iq[current_index] = np.fft.fftshift(iq[current_index])
                iq[current_index] = self.normalize(iq=iq[current_index])

        return iq

    def generate_pca_csv_data(self, csv_labels, validation_indices):
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

    def dump_PCA_data(self, pickle_name, csv_name, validation_indices):
        os.chdir(self.dataset_dir)
        with open('{}'.format(pickle_name), 'rb') as pickle_file:
            raw_data = pickle.load(pickle_file)
            self.analyse_PCA_hyperparameters(iq_list=raw_data["iq_sweep_burst"][0:100])
            iq = self.generate_pca_expansion_for_iq_samples(pkl_raw_data=raw_data, validation_indices=validation_indices)

        with open('{}'.format(csv_name)) as csv_file:
            labels = csv.reader(csv_file, delimiter=',', quotechar='|')
            labels_dict = self.generate_pca_csv_data(csv_labels=labels, validation_indices=validation_indices)

        labels_dict['target_type'] = [0 if y == 'animal' else 1 for y in labels_dict['target_type']]

        return iq, labels_dict

    '''
    PCA Synthetic Augmentation
    '''
    def generate_pca_synthetic_expansion_for_iq_samples(self, pkl_raw_data, required_segments):
        doppler_burst = pkl_raw_data['doppler_burst']
        raw_iq_matrices = pkl_raw_data['iq_sweep_burst']
        raw_iq_segments = pkl_raw_data['segment_id'].tolist()
        expansion_size = self.config.num_of_pca_segments
        train_segments_id = [int(seg) for seg in required_segments]

        if self.config.with_iq_matrices is True or self.config.with_magnitude_phase is True:
            iq = np.zeros((expansion_size * len(required_segments), raw_iq_matrices.shape[1] - 2, raw_iq_matrices.shape[2],2))
        else:
            iq = np.zeros((expansion_size * len(required_segments), raw_iq_matrices.shape[1] - 2, raw_iq_matrices.shape[2]))

        for segment, segment_id in enumerate(train_segments_id):
            idx = raw_iq_segments.index(segment_id)
            for pca_index, current_iq_mat in zip(range(expansion_size), self.PCA_time_augmentation(raw_iq_matrices[idx])):
                current_index = self.config.num_of_pca_segments * segment + pca_index
                iq[current_index] = self.fft(current_iq_mat)
                if self.config.with_max_value_on_doppler is True:
                    iq[current_index] = self.max_value_on_doppler(iq=iq[current_index], doppler_burst=doppler_burst[idx])
                else:
                    iq[current_index] = np.fft.fftshift(iq[current_index])
                iq[current_index] = self.normalize(iq=iq[current_index])
        return iq

    def generate_pca_synthetic_csv_data(self, csv_labels, required_segments):
        is_first_line = True
        labels_augmented = collections.OrderedDict()

        for line in csv_labels:
            if is_first_line:
                is_first_line = False
                for key in line:
                    labels_augmented[key] = []
            else:
                line_is_required_data = False

                # Check if this line contains a data this is a part of the validation data set
                for key, data in zip(labels_augmented.keys(), line):
                    if key == 'segment_id':
                        if data in required_segments:
                            line_is_required_data = True
                        break

                if line_is_required_data is True:
                    for key, data in zip(labels_augmented.keys(), line):
                        for index in range(self.config.num_of_pca_segments):
                            # Duplicate the data
                            if key == 'segment_id':
                                labels_augmented[key].append(str(-int(data)))
                            else:
                                labels_augmented[key].append(data)

        return labels_augmented

    def dump_PCA_synthetic_data(self, pickle_name, csv_name, required_segments):
        os.chdir(self.dataset_dir)
        with open('{}'.format(pickle_name), 'rb') as pickle_file:
            raw_data = pickle.load(pickle_file)
            iq = self.generate_pca_synthetic_expansion_for_iq_samples(pkl_raw_data=raw_data, required_segments=required_segments)

        with open('{}'.format(csv_name)) as csv_file:
            labels = csv.reader(csv_file, delimiter=',', quotechar='|')
            labels_dict = self.generate_pca_synthetic_csv_data(csv_labels=labels, required_segments=required_segments)

        labels_dict['target_type'] = [0 if y == 'animal' else 1 for y in labels_dict['target_type']]

        return iq, labels_dict


    '''
    Time shift using random phase multiplication expansion for train dataset
    '''
    def phase_time_shift_augmentation(self, iq):
        '''
        Calculate the PCA of the vector and generate samples from it
        '''

        '''
        Generate new samples
        '''
        # SLOW_VARIANCE_SCALING = self.config.phase_distortion_slow_axis_variance
        FAST_VARIANCE_SCALING = self.config.phase_time_shift_fast_axis_variance
        X_augment = []
        # slow_axis_vec = np.linspace(start=0,stop=iq.shape[1]-1,num=iq.shape[1])
        fast_axis_vec = np.linspace(start=0,stop=iq.shape[0]-1,num=iq.shape[0])
        for test_index in range(self.config.num_of_phase_time_shift):
            phase_matrix = np.transpose(np.array([np.exp(2j*np.pi*fast_axis_vec*np.random.randn(1)*FAST_VARIANCE_SCALING/iq.shape[0]) for x in range(iq.shape[1])]))
            new_iq_mat = np.multiply(iq, phase_matrix)
            X_augment.append(new_iq_mat)


        # figure, axes = plt.subplots(nrows=2, ncols=2)
        # axes[0, 0].imshow(self.normalize(self.fft(iq)))
        # axes[0, 0].set_title('Freq Response Org')
        # axes[0, 1].imshow(self.normalize(self.fft(X_augment[0])))
        # axes[0, 1].set_title('Freq Response Generated 1 ')
        # axes[1, 0].imshow(self.normalize(self.fft(X_augment[1])))
        # axes[1, 0].set_title('Freq Response Generated 2 ')
        # axes[1, 1].imshow(self.normalize(self.fft(X_augment[2])))
        # axes[1, 1].set_title('Freq Response Generated 3 ')
        # figure.savefig('test_phase_distortion')
        #
        # plt.figure()
        # plt.imshow(self.normalize(self.fft(iq)))
        # plt.savefig('Org IQ')
        #
        # for x,index in zip(X_augment,range(len(X_augment))):
        #     plt.figure()
        #     plt.imshow(self.normalize(self.fft(x)))
        #     print('error for index {} is {}'.format(index,np.linalg.norm(np.cov(iq) - np.cov(X_augment[index])) / np.linalg.norm(np.cov(iq))))
        #     plt.savefig('Phase Augmentated IQ{}'.format(index))

        return X_augment

    def dump_time_shift_using_phase_data(self,pickle_name,csv_name,validation_indices):
        os.chdir(self.dataset_dir)
        with open('{}'.format(pickle_name), 'rb') as pickle_file:
            raw_data = pickle.load(pickle_file)
            iq = self.generate_time_shift_using_phase_data(pkl_raw_data=raw_data, validation_indices=validation_indices)

        with open('{}'.format(csv_name)) as csv_file:
            labels = csv.reader(csv_file, delimiter=',', quotechar='|')
            labels_dict = self.generate_time_shift_using_phase_csv_data(csv_labels=labels, validation_indices=validation_indices)

        labels_dict['target_type'] = [0 if y == 'animal' else 1 for y in labels_dict['target_type']]

        return iq, labels_dict

    def generate_time_shift_using_phase_data(self,pkl_raw_data,validation_indices):
        doppler_burst = pkl_raw_data['doppler_burst']
        raw_iq_matrices = pkl_raw_data['iq_sweep_burst']
        raw_iq_segments = pkl_raw_data['segment_id'].tolist()

        expansion_size = self.config.num_of_phase_time_shift
        train_segments_id = [segment_id for segment_id in raw_iq_segments if str(segment_id) not in validation_indices]  # remove the validation indices

        if self.config.with_iq_matrices is True or self.config.with_magnitude_phase is True:
            iq = np.zeros((expansion_size * len(train_segments_id), raw_iq_matrices.shape[1] - 2, raw_iq_matrices.shape[2], 2))
        else:
            iq = np.zeros((expansion_size * len(train_segments_id), raw_iq_matrices.shape[1] - 2, raw_iq_matrices.shape[2]))

        for segment, segment_id in enumerate(train_segments_id):
            idx = raw_iq_segments.index(segment_id)
            for cyclic_index, current_iq_mat in zip(range(expansion_size), self.phase_time_shift_augmentation(raw_iq_matrices[idx])):
                current_index = self.config.num_of_phase_time_shift * segment + cyclic_index
                iq[current_index] = self.fft(current_iq_mat)
                if self.config.with_max_value_on_doppler is True:
                    iq[current_index] = self.max_value_on_doppler(iq=iq[current_index], doppler_burst=doppler_burst[idx])
                else:
                    iq[current_index] = np.fft.fftshift(iq[current_index])
                iq[current_index] = self.normalize(iq=iq[current_index])
        return iq

    def generate_time_shift_using_phase_csv_data(self, csv_labels, validation_indices):
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
                        for index in range(self.config.num_of_phase_time_shift):
                            # Duplicate the data
                            if key == 'segment_id':
                                labels_augmented[key].append(str(-int(data)))
                            else:
                                labels_augmented[key].append(data)

        return labels_augmented

    '''
    Time shift using random phase multiplication for synthetic data set
    '''
    def generate_phase_time_shift_synthetic_expansion_for_iq_samples(self, pkl_raw_data, required_segments):
        doppler_burst = pkl_raw_data['doppler_burst']
        raw_iq_matrices = pkl_raw_data['iq_sweep_burst']
        raw_iq_segments = pkl_raw_data['segment_id'].tolist()
        expansion_size = self.config.num_of_phase_time_shift
        train_segments_id = [int(seg) for seg in required_segments]

        if self.config.with_iq_matrices is True or self.config.with_magnitude_phase is True:
            iq = np.zeros((expansion_size * len(required_segments), raw_iq_matrices.shape[1] - 2, raw_iq_matrices.shape[2], 2))
        else:
            iq = np.zeros((expansion_size * len(required_segments), raw_iq_matrices.shape[1] - 2, raw_iq_matrices.shape[2]))

        for segment, segment_id in enumerate(train_segments_id):
            idx = raw_iq_segments.index(segment_id)
            for cyclic_index, current_iq_mat in zip(range(expansion_size), self.phase_time_shift_augmentation(raw_iq_matrices[idx])):
                current_index = self.config.num_of_phase_time_shift * segment + cyclic_index
                iq[current_index] = self.fft(current_iq_mat)
                if self.config.with_max_value_on_doppler is True:
                    iq[current_index] = self.max_value_on_doppler(iq=iq[current_index], doppler_burst=doppler_burst[idx])
                else:
                    iq[current_index] = np.fft.fftshift(iq[current_index])
                iq[current_index] = self.normalize(iq=iq[current_index])
        return iq

    def generate_phase_time_shift_synthetic_csv_data(self, csv_labels, required_segments):
        is_first_line = True
        labels_augmented = collections.OrderedDict()

        for line in csv_labels:
            if is_first_line:
                is_first_line = False
                for key in line:
                    labels_augmented[key] = []
            else:
                line_is_required_data = False

                # Check if this line contains a data this is a part of the validation data set
                for key, data in zip(labels_augmented.keys(), line):
                    if key == 'segment_id':
                        if data in required_segments:
                            line_is_required_data = True
                        break

                if line_is_required_data is True:
                    for key, data in zip(labels_augmented.keys(), line):
                        for index in range(self.config.num_of_phase_time_shift):
                            # Duplicate the data
                            if key == 'segment_id':
                                labels_augmented[key].append(str(-int(data)))
                            else:
                                labels_augmented[key].append(data)

        return labels_augmented

    def dump_phase_time_shift_synthetic_data(self, pickle_name, csv_name, required_segments):
        os.chdir(self.dataset_dir)
        with open('{}'.format(pickle_name), 'rb') as pickle_file:
            raw_data = pickle.load(pickle_file)
            iq = self.generate_phase_time_shift_synthetic_expansion_for_iq_samples(pkl_raw_data=raw_data, required_segments=required_segments)

        with open('{}'.format(csv_name)) as csv_file:
            labels = csv.reader(csv_file, delimiter=',', quotechar='|')
            labels_dict = self.generate_phase_time_shift_synthetic_csv_data(csv_labels=labels, required_segments=required_segments)

        labels_dict['target_type'] = [0 if y == 'animal' else 1 for y in labels_dict['target_type']]

        return iq, labels_dict


    '''
    Window Augmentation
    '''
    def generate_window_data(self,pkl_raw_data,validation_indices):
        doppler_burst = pkl_raw_data['doppler_burst']
        raw_iq_matrices = pkl_raw_data['iq_sweep_burst']
        raw_iq_segments = pkl_raw_data['segment_id'].tolist()

        train_segments_id = [segment_id for segment_id in raw_iq_segments if str(segment_id) not in validation_indices]  # remove the validation indices

        if self.config.with_iq_matrices is True or self.config.with_magnitude_phase is True:
            iq = np.zeros((len(train_segments_id), raw_iq_matrices.shape[1] - 2, raw_iq_matrices.shape[2], 2))
        else:
            iq = np.zeros((len(train_segments_id), raw_iq_matrices.shape[1] - 2, raw_iq_matrices.shape[2]))

        for index, segment_id in enumerate(train_segments_id):
            idx = raw_iq_segments.index(segment_id)
            iq[index] = self.fft(raw_iq_matrices[idx],window_type=self.config.default_window_type)
            if self.config.with_max_value_on_doppler is True:
                iq[index] = self.max_value_on_doppler(iq=iq[index], doppler_burst=doppler_burst[idx])
            else:
                iq[index] = np.fft.fftshift(iq[index])
            iq[index] = self.normalize(iq=iq[index])

        return iq

    def generate_window_csv_data(self, csv_labels, validation_indices):
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
                        if key == 'segment_id':
                            labels_augmented[key].append(str(-int(data)))
                        else:
                            labels_augmented[key].append(data)

        return labels_augmented

    def dump_window_augmentation(self,window_type,pickle_name, csv_name,validation_indices):
        self.config.__setattr__("default_window_type",window_type)

        os.chdir(self.dataset_dir)
        with open('{}'.format(pickle_name), 'rb') as pickle_file:
            raw_data = pickle.load(pickle_file)
            iq = self.generate_window_data(pkl_raw_data=raw_data,validation_indices=validation_indices)

        with open('{}'.format(csv_name)) as csv_file:
            labels = csv.reader(csv_file, delimiter=',', quotechar='|')
            labels_dict = self.generate_window_csv_data(csv_labels=labels,validation_indices=validation_indices)

        self.config.__setattr__("default_window_type", "hann")
        return iq, labels_dict

    '''
    Window Synthetic Augmentation
    '''
    def generate_window_synthetic_expansion_for_iq_samples(self, pkl_raw_data, required_segments):
        doppler_burst = pkl_raw_data['doppler_burst']
        raw_iq_matrices = pkl_raw_data['iq_sweep_burst']
        raw_iq_segments = pkl_raw_data['segment_id'].tolist()

        train_segments_id = [int(seg) for seg in required_segments]

        if self.config.with_iq_matrices is True or self.config.with_magnitude_phase is True:
            iq = np.zeros((len(required_segments), raw_iq_matrices.shape[1] - 2, raw_iq_matrices.shape[2],2))
        else:
            iq = np.zeros((len(required_segments), raw_iq_matrices.shape[1] - 2, raw_iq_matrices.shape[2]))

        for index, segment_id in enumerate(train_segments_id):
            idx = raw_iq_segments.index(segment_id)
            iq[index] = self.fft(raw_iq_segments[idx],window_type=self.config.default_window_type)
            if self.config.with_max_value_on_doppler is True:
                iq[index] = self.max_value_on_doppler(iq=iq[index], doppler_burst=doppler_burst[idx])
            else:
                iq[index] = np.fft.fftshift(iq[index])
            iq[index] = self.normalize(iq=iq[index])
        return iq

    def generate_window_synthetic_csv_data(self, csv_labels, required_segments):
        is_first_line = True
        labels_augmented = collections.OrderedDict()

        for line in csv_labels:
            if is_first_line:
                is_first_line = False
                for key in line:
                    labels_augmented[key] = []
            else:
                line_is_required_data = False

                # Check if this line contains a data this is a part of the validation data set
                for key, data in zip(labels_augmented.keys(), line):
                    if key == 'segment_id':
                        if data in required_segments:
                            line_is_required_data = True
                        break

                if line_is_required_data is True:
                    for key, data in zip(labels_augmented.keys(), line):
                        if key == 'segment_id':
                            labels_augmented[key].append(str(-int(data)))
                        else:
                            labels_augmented[key].append(data)

        return labels_augmented

    def dump_window_synthetic_data(self, window_type, pickle_name, csv_name, required_segments):
        self.config.__setattr__("default_window_type", window_type)

        os.chdir(self.dataset_dir)
        with open('{}'.format(pickle_name), 'rb') as pickle_file:
            raw_data = pickle.load(pickle_file)
            iq = self.generate_window_synthetic_expansion_for_iq_samples(pkl_raw_data=raw_data, required_segments=required_segments)

        with open('{}'.format(csv_name)) as csv_file:
            labels = csv.reader(csv_file, delimiter=',', quotechar='|')
            labels_dict = self.generate_window_synthetic_csv_data(csv_labels=labels, required_segments=required_segments)

        labels_dict['target_type'] = [0 if y == 'animal' else 1 for y in labels_dict['target_type']]

        self.config.__setattr__("default_window_type", "hann")

        return iq, labels_dict




