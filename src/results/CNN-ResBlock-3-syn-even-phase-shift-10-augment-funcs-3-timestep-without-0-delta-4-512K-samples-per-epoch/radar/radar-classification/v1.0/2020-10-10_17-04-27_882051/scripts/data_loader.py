import copy
import tensorflow as tf
import tensorflow_datasets as tfds
from data.data_parser import *
from tensorflow.keras.utils import to_categorical
from data.data_loader_utils import *
from scipy.ndimage import gaussian_filter
from data.signal_processing import *
import pickle
import re
from scipy.signal import resample, resample_poly


# def generator(data, config):
#     batch_features = np.zeros((config.batch_size, config.rect_augment_num_of_rows, config.rect_augment_num_of_cols, 1))
#     batch_labels = np.zeros((config.batch_size, 1))
#
#     while True:
#         for batch_X, batch_y in data:
#             X, y = expand_data_generator_by_sampling_rect(batch_X, batch_y, config)
#             array_indices = np.arange(len(X))
#             for batch_repeat_index in range(config.repeat_rect_exp_per_batch):
#                 for i in range(config.batch_size):
#                     # choose random index in features
#                     index = np.random.choice(array_indices, 1)[0]
#                     # Delete index from the future options
#                     array_indices = np.delete(array_indices, np.where(index == array_indices)[0][0])
#                     # Crop the image
#                     batch_features[i] = X[index]
#                     batch_labels[i] = y[index]
#             yield (batch_features, batch_labels)

def generate_test_set(train_data, TRACK_ID_TEST_SET_SIZE, snr_type):
    """
    Since test date given is un-labeled, in order to evaluate the the model properly we need to hold-out training
    examples from the train set to be used as test.
    The train set is divided into tracks so we need to make sure there are no segments from same tracks in train & test
    to avoid inherited over-fitting.
    """
    global X_valid_low, X_valid_high, indices_low, indices_high, X_train_high, X_train_low, labels_valid_high, labels_valid_low, labels_train_high, labels_train_low
    train_data_orig = train_data
    assert snr_type == 'all' or snr_type == 'high' or snr_type == 'low'

    # HighSNR
    if snr_type == 'high' or snr_type == 'all':
        count_valid_human = TRACK_ID_TEST_SET_SIZE // 2
        count_valid_animal = TRACK_ID_TEST_SET_SIZE // 2 if snr_type == 'high' else TRACK_ID_TEST_SET_SIZE // 4
        X_valid_high, labels_valid_high, X_train_high, labels_train_high, indices_high = \
            split_train_and_valid_by_snr(snr_type='HighSNR', X=train_data_orig[0], labels=train_data_orig[1],
                                         count_valid_animal=count_valid_animal,
                                         count_valid_human=count_valid_human, all_snr=True if snr_type == 'all' else False)
        if snr_type != 'all':
            train_data = (X_train_high, labels_train_high)
            validation_data = (X_valid_high, labels_valid_high)

    # 'LowSNR'
    if snr_type == 'low' or snr_type == 'all':
        count_valid_human = TRACK_ID_TEST_SET_SIZE // 2 if snr_type == 'low' else TRACK_ID_TEST_SET_SIZE // 4
        count_valid_animal = TRACK_ID_TEST_SET_SIZE // 2 if snr_type == 'low' else TRACK_ID_TEST_SET_SIZE // 4
        X_valid_low, labels_valid_low, X_train_low, labels_train_low, indices_low = \
            split_train_and_valid_by_snr(snr_type='LowSNR', X=train_data_orig[0], labels=train_data_orig[1],
                                         count_valid_animal=count_valid_animal,
                                         count_valid_human=count_valid_human, all_snr=True if snr_type == 'all' else False)
        if snr_type != 'all':
            train_data = (X_train_low, labels_train_low)
            validation_data = (X_valid_low, labels_valid_low)

    # 'AllSNR'
    if snr_type == 'all':
        assert not (True in (indices_low & indices_high))
        # mixed indices - of tracks with both high and low snr
        # high_indices, low_indices BOTH are arrays that have no duality snr
        mix_idx = np.bool_(np.ones(len(indices_high))) & np.logical_not(indices_high | indices_low)
        X_train_mixed = train_data_orig[0][mix_idx]
        labels_train_mixed = collections.OrderedDict()
        for key in train_data_orig[1].keys():
            labels_train_mixed[key] = (np.array(train_data_orig[1][key])[mix_idx]).tolist()

        X_valid = np.concatenate((X_valid_high, X_valid_low), axis=0)
        X_train = np.concatenate((X_train_high, X_train_low, X_train_mixed), axis=0)
        labels_valid = collections.OrderedDict()
        labels_train = collections.OrderedDict()
        for key in train_data_orig[1].keys():
            # validation
            labels_valid[key] = labels_valid_high[key]
            labels_valid[key].extend(labels_valid_low[key])
            # train
            labels_train[key] = labels_train_high[key]
            labels_train[key].extend(labels_train_low[key])
            labels_train[key].extend(labels_train_mixed[key])
        train_data = (X_train, labels_train)
        validation_data = (X_valid, labels_valid)
        assert len(train_data[0]) + len(validation_data[0]) == len(train_data_orig[0])

    return train_data, validation_data

    # TODO: set here validation according to the sort and the fact there are chances that we are sampling the array

    # sorted_segments_id = np.sort(np.array([int(numeric_string) for numeric_string in labels_train['segment_id']]))
    #
    # for segment_num,new_index in zip(sorted_segments_id,range(sorted_segments_id.shape[0])):
    #     labels_train['segment_id'][labels_train['segment_id'].index(str(segment_num))] = str(new_index)
    # for i in range(X_train.shape[0]):
    #     print(int(labels_train['segment_id'][i]))
    #     if np.linalg.norm(X_train[i] - train_data_orig[0][int(labels_train['segment_id'][i])]) != 0:
    #         raise Exception('sorting missmatch! , at X_train[{}]'.format(i))
    # for i in range(X_valid.shape[0]):
    #     if np.linalg.norm(X_valid[i] - train_data_orig[0][int(labels_valid['segment_id'][i])]) != 0:
    #         raise Exception('sorting missmatch! , at X_valid[{}]'.format(i))


def expand_data_by_pca_time_comperssion(data_parser, train_data, validation_segments_id_list):
    # Validate that the interpolated data set has no bias with the validation set
    augmentad_data = data_parser.dump_PCA_data(pickle_name='MAFAT RADAR Challenge - Training Set V1.pkl',
                                               csv_name='MAFAT RADAR Challenge - Training Set V1.csv',
                                               validation_indices=validation_segments_id_list, method='compression')

    for segment_id in augmentad_data[1]['segment_id']:
        if segment_id[1:] in validation_segments_id_list:
            raise Exception('Using PCA augmentation and data is biased with the validation data set !!')
    X = np.concatenate((train_data[0], augmentad_data[0]), axis=0)  # Stack the interpolated data
    train_labels = train_data[1]
    train_augment_labels = augmentad_data[1]
    for key in train_labels.keys():
        train_labels[key].extend(train_augment_labels[key])
    labels = train_labels

    return X, labels


def expand_data_by_pca_time_augmentation(data_parser, train_data, validation_segments_id_list):
    # Validate that the interpolated data set has no bias with the validation set
    augmentad_data = data_parser.dump_PCA_data(pickle_name='MAFAT RADAR Challenge - Training Set V1.pkl',
                                               csv_name='MAFAT RADAR Challenge - Training Set V1.csv',
                                               validation_indices=validation_segments_id_list)

    for segment_id in augmentad_data[1]['segment_id']:
        if segment_id[1:] in validation_segments_id_list:
            raise Exception('Using PCA augmentation and data is biased with the validation data set !!')
    X = np.concatenate((train_data[0], augmentad_data[0]), axis=0)  # Stack the interpolated data
    train_labels = train_data[1]
    train_augment_labels = augmentad_data[1]
    for key in train_labels.keys():
        train_labels[key].extend(train_augment_labels[key])
    labels = train_labels

    return X, labels


def transform_data_with_ZCA(train_data):
    X = train_data[0]
    labels = train_data[1]
    X = ZCA_transform(X)
    return X, labels


def expand_data_by_freq_rotation(data_parser, train_data, validation_segments_id_list):
    augmentad_data = data_parser.dump_freq_rotation_data(pickle_name='MAFAT RADAR Challenge - Training Set V1.pkl',
                                                         csv_name='MAFAT RADAR Challenge - Training Set V1.csv',
                                                         validation_indices=validation_segments_id_list)
    for segment_id in augmentad_data[1]['segment_id']:
        if segment_id[1:] in validation_segments_id_list:
            raise Exception('Using phase distortion augmentation and data is biased with the validation data set !!')
    X = np.concatenate((train_data[0], augmentad_data[0]), axis=0)  # Stack the interpolated data
    train_labels = train_data[1]
    train_augment_labels = augmentad_data[1]
    for key in train_labels.keys():
        train_labels[key].extend(train_augment_labels[key])
    labels = train_labels

    return X, labels


def expand_data_by_phase_time_shift(data_parser, train_data, validation_segments_id_list):
    # Validate that the interpolated data set has no bias with the validation set

    augmentad_data = data_parser.dump_time_shift_using_phase_data(pickle_name='MAFAT RADAR Challenge - Training Set V1.pkl',
                                                                  csv_name='MAFAT RADAR Challenge - Training Set V1.csv',
                                                                  validation_indices=validation_segments_id_list)

    for segment_id in augmentad_data[1]['segment_id']:
        if segment_id[1:] in validation_segments_id_list:
            raise Exception('Using phase distortion augmentation and data is biased with the validation data set !!')
    X = np.concatenate((train_data[0], augmentad_data[0]), axis=0)  # Stack the interpolated data
    train_labels = train_data[1]
    train_augment_labels = augmentad_data[1]
    for key in train_labels.keys():
        train_labels[key].extend(train_augment_labels[key])
    labels = train_labels

    return X, labels


def expand_data_by_window_function(data_parser, train_data, validation_segments_id_list, config):
    # Validate that the interpolated data set has no bias with the validation set
    if len(config.window_list) == 0:
        raise Exception('Can\'t use window augmentation with empty window list')

    for window in config.window_list:
        augmentad_data = data_parser.dump_window_augmentation(window_type=window, pickle_name='MAFAT RADAR Challenge - Training Set V1.pkl',
                                                              csv_name='MAFAT RADAR Challenge - Training Set V1.csv',
                                                              validation_indices=validation_segments_id_list)

        for segment_id in augmentad_data[1]['segment_id']:
            if segment_id[1:] in validation_segments_id_list:
                raise Exception('Using phase distortion augmentation and data is biased with the validation data set !!')
        X = np.concatenate((train_data[0], augmentad_data[0]), axis=0)  # Stack the interpolated data
        train_labels = train_data[1]
        train_augment_labels = augmentad_data[1]
        for key in train_labels.keys():
            train_labels[key].extend(train_augment_labels[key])
        labels = train_labels

    return X, labels


def expand_human_data_by_tracks(train_data, config):
    """
    histogram of human tracks
    list1 = [len(np.array([i for i, x in enumerate(tid_int_list) if x == tid])) for tid in tid_unique_list]
    plt.figure()
    plt.hist(list1,bins = [x for x in range(max(list1))])
    plt.savefig('../hist1')
    """

    # sort train_data by track_id
    X_sorted, labels_sorted = sort_data_by_track_id(train_data)
    idx_human = np.array([i for i in range(len(X_sorted)) if labels_sorted['target_type'][i] == 1])
    X_sorted_human = X_sorted[idx_human]
    labels_sorted_human = collections.OrderedDict()
    for key in labels_sorted.keys():
        labels_sorted_human[key] = np.array(labels_sorted[key])[idx_human].tolist()

    # remove animal data
    X_augmented = []
    augment_count = 0
    labels_augmented = collections.OrderedDict()
    for key in labels_sorted_human.keys():
        labels_augmented[key] = []
    offset = config.augment_by_track_offset
    tid_int_list = [int(tid) for tid in labels_sorted_human['track_id']]
    tid_unique_list = np.unique(np.array(tid_int_list)).tolist()
    skip_counter = 0
    #
    for tid in tid_unique_list:
        i_list = np.array([i for i, x in enumerate(tid_int_list) if x == tid])
        assert labels_sorted_human['target_type'][i_list[0]] == 1
        if len(i_list) < 2:
            skip_counter = skip_counter + 1
            continue
        X_tid = np.concatenate(X_sorted_human[i_list], axis=1)
        offset_list = [offset + i * 32 for i in range(len(i_list)) if offset + i * 32 < (len(i_list) - 1) * 32]
        indices = np.random.choice(offset_list, size=min(len(offset_list), config.augment_by_track_local_count),
                                   replace=False).tolist()
        for i in indices:
            X_augmented.append(X_tid[:, i:i + 32])
            for key in labels_augmented.keys():
                if key == 'segment_id':
                    labels_augmented[key].append(str(-(augment_count + 1)))
                else:
                    labels_augmented[key].append(labels_sorted_human[key][i_list[0]])
            augment_count = augment_count + 1

    print('augment_by_track addition: {}'.format(augment_count))
    X_augmented = np.stack(X_augmented, axis=0)
    X_train_new = np.concatenate((X_sorted, X_augmented), axis=0)
    labels_new = collections.OrderedDict()
    for key in labels_sorted.keys():
        labels_new[key] = labels_sorted[key]
        labels_new[key].extend(labels_augmented[key])

    return X_train_new, labels_new


def expand_data_generator_by_sampling_rect(X, y, config):
    new_data = []
    new_label = []
    for segment, index in zip(X, range(len(X))):
        expanded_iq = Sample_rectangle_from_spectrogram(iq_mat=segment, config=config)
        new_data.extend(expanded_iq)
        new_label_list = [y[index]] * len(expanded_iq)
        new_label.extend(new_label_list)

    return new_data, new_label


def expand_data_by_sampling_rect(data, config):
    new_data = []
    new_labels_dict = collections.OrderedDict()
    for key in data[1].keys():
        new_labels_dict[key] = []

    for segment, index in zip(data[0], range(len(data[0]))):
        expanded_iq = Sample_rectangle_from_spectrogram(iq_mat=segment, config=config)

        sampled_indices = np.random.choice(len(expanded_iq),
                                           size=config.sample_for_segment_rect_augmentation,
                                           replace=False)
        sampled_iq_list = [expanded_iq[index] for index in sampled_indices]
        new_data.extend(sampled_iq_list)
        for key in new_labels_dict.keys():
            new_label_list = [data[1][key][index]] * len(sampled_iq_list)
            new_labels_dict[key].extend(new_label_list)

    new_data = np.array(new_data)
    data = (new_data, new_labels_dict)
    return data


def expand_test_by_sampling_rect(data, config):
    new_data = []

    for segment, index in zip(data, range(len(data))):
        expanded_iq = Sample_rectangle_from_spectrogram(iq_mat=segment, config=config)
        sampled_indices = np.random.choice(len(expanded_iq),
                                           size=min(config.sample_for_test_segment_rect_augmentation, len(expanded_iq)),
                                           replace=False)
        sampled_iq_list = [expanded_iq[index] for index in sampled_indices]
        new_data.append(sampled_iq_list)

    return new_data


def expand_human_data_by_aux_set(data_parser, train_data):
    num_of_animals = sum([1 for target in train_data[1]['target_type'] if target == 0])
    num_of_humans = sum([1 for target in train_data[1]['target_type'] if target == 1])
    # get complementary human examples from synthetic set
    train_data_aux_exp = data_parser.aux_split(num_of_animals, num_of_humans)
    X = np.concatenate((train_data[0], train_data_aux_exp[0]), axis=0)  # Stack the experiment data
    train_labels = train_data[1]
    train_aux_exp_labels = train_data_aux_exp[1]
    for key in train_labels.keys():
        train_labels[key].extend(train_aux_exp_labels[key])
    labels = train_labels
    # validate 50/50 distribution
    num_of_animals = sum([1 for target in train_data[1]['target_type'] if target == 0])
    num_of_humans = sum([1 for target in train_data[1]['target_type'] if target == 1])
    if abs(num_of_animals / (num_of_animals + num_of_humans) - 0.5) > 0.15:
        raise Exception('Using stable mode and data resulted as unstable !!')
    return X, labels


def expand_data_by_human_low_exp_set(data_parser, train_data, config):
    aux_exp_data = data_parser.get_dataset_by_snr(dataset_type='aux_exp', snr_type=config.snr_type)
    aux_exp_snr_type = np.array(aux_exp_data[1]['snr_type'])
    idx_low_list = np.where(aux_exp_snr_type == 'LowSNR')[0].tolist()
    idx_low = np.bool_(np.zeros(len(aux_exp_data[0])))
    idx_low[idx_low_list] = True

    X = np.concatenate((train_data[0], aux_exp_data[0][idx_low]), axis=0)  # Stack the experiment data
    train_labels = train_data[1]

    for key in train_labels.keys():
        train_labels[key].extend(np.array(aux_exp_data[1][key])[idx_low].tolist())
    labels = train_labels

    if config.with_pca_augmentation:
        required_indices = np.where(idx_low == True)[0].tolist()
        segid_arr = np.array(aux_exp_data[1]['segment_id'])
        required_segments = segid_arr[required_indices].tolist()
        exp_pca_data = data_parser.dump_PCA_synthetic_data(
            pickle_name='MAFAT RADAR Challenge - Auxiliary Experiment Set V2.pkl',
            csv_name='MAFAT RADAR Challenge - Auxiliary Experiment Set V2.csv', required_segments=required_segments, expansion_method='augmentation')
        X = np.concatenate((X, exp_pca_data[0]), axis=0)
        for key in train_data[1].keys():
            train_data[1][key].extend(exp_pca_data[1][key])
        labels = train_data[1]
    if config.with_pca_compression:
        required_indices = np.where(idx_low == True)[0].tolist()
        segid_arr = np.array(aux_exp_data[1]['segment_id'])
        required_segments = segid_arr[required_indices].tolist()
        exp_pca_data = data_parser.dump_PCA_synthetic_data(
            pickle_name='MAFAT RADAR Challenge - Auxiliary Experiment Set V2.pkl',
            csv_name='MAFAT RADAR Challenge - Auxiliary Experiment Set V2.csv', required_segments=required_segments, expansion_method='compression')
        X = np.concatenate((X, exp_pca_data[0]), axis=0)
        for key in train_data[1].keys():
            train_data[1][key].extend(exp_pca_data[1][key])
        labels = train_data[1]
    if config.with_phase_time_shift:
        required_indices = np.where(idx_low == True)[0].tolist()
        segid_arr = np.array(aux_exp_data[1]['segment_id'])
        required_segments = segid_arr[required_indices].tolist()
        exp_phase_data = data_parser.dump_phase_time_shift_synthetic_data(
            pickle_name='MAFAT RADAR Challenge - Auxiliary Experiment Set V2.pkl',
            csv_name='MAFAT RADAR Challenge - Auxiliary Experiment Set V2.csv', required_segments=required_segments, expansion_method='augmentation')
        X = np.concatenate((X, exp_phase_data[0]), axis=0)
        for key in train_data[1].keys():
            train_data[1][key].extend(exp_phase_data[1][key])
        labels = train_data[1]

    if config.with_window_augmentation:
        required_indices = np.where(idx_low == True)[0].tolist()
        segid_arr = np.array(aux_exp_data[1]['segment_id'])
        required_segments = segid_arr[required_indices].tolist()
        if len(config.window_list) == 0:
            raise Exception('Can\'t use window augmentation without window list')
        for window in config.window_list:

            exp_phase_data = data_parser.dump_window_synthetic_data(window_type=window,
                                                                    pickle_name='MAFAT RADAR Challenge - Auxiliary Experiment Set V2.pkl',
                                                                    csv_name='MAFAT RADAR Challenge - Auxiliary Experiment Set V2.csv',
                                                                    required_segments=required_segments)
            X = np.concatenate((X, exp_phase_data[0]), axis=0)
            for key in train_data[1].keys():
                train_data[1][key].extend(exp_phase_data[1][key])
            labels = train_data[1]

    return X, labels


def expand_data_by_synthetic_set(data_parser, train_data, config, validation_data):
    aux_syn_data = data_parser.get_dataset_by_snr(dataset_type='aux_syn', snr_type=config.snr_type)
    aux_syn_tracks = np.array(aux_syn_data[1]['track_id'])
    train_tracks = np.array(train_data[1]['track_id'])
    aux_syn_target = np.array(aux_syn_data[1]['target_type'])
    validation_segments = [int(segid) for segid in validation_data[1]['segment_id']]

    # concatenate all animal segments from synthetic set since it is with mostly human
    # idx_animal_debug = np.where(aux_syn_target == 0)[0].tolist()
    idx_animal_list = np.where(aux_syn_target == 0)[0].tolist()
    idx_animal_list = [idx for idx in idx_animal_list if int(aux_syn_data[1]['segment_id'][idx]) - 2000000 not in validation_segments]
    idx_animal = np.bool_(np.zeros(len(aux_syn_data[0])))
    idx_animal[idx_animal_list] = True
    X = np.concatenate((train_data[0], aux_syn_data[0][idx_animal]), axis=0)
    for key in train_data[1].keys():
        aux_syn_arr = np.array(aux_syn_data[1][key])
        train_data[1][key].extend(aux_syn_arr[idx_animal].tolist())

    # aux_syn_tracks_unique_list = unique list of tracks in synthetic set that do not already appear in train and are not of animal target
    aux_syn_tracks_unique = np.unique(aux_syn_tracks)
    train_tracks_unique = np.unique(train_tracks)
    aux_syn_tracks_unique_list = [tid for tid in aux_syn_tracks_unique.tolist() if
                                  tid not in train_tracks_unique.tolist()]
    assert not (False in [tid in aux_syn_data[1]['track_id'] for tid in aux_syn_tracks_unique_list])
    # aux_syn_tracks_unique_list_debug = [tid for tid in aux_syn_tracks_unique_list if
    #                                     aux_syn_data[1]['target_type'][aux_syn_tracks_unique_list.index(tid)] != 0]
    aux_syn_tracks_unique_list = [tid for tid in aux_syn_tracks_unique_list if
                                  aux_syn_data[1]['target_type'][np.where(aux_syn_tracks == tid)[0][0]] != 0]
    assert not (True in [True if aux_syn_data[1]['target_type'][np.where(aux_syn_tracks == tid)[0][0]] == 0 else False
                         for tid in aux_syn_tracks_unique_list])

    # count ratio of expansion: r = (N_A - N_H) / N_unique_tracks
    target_train = np.array(train_data[1]['target_type'])
    N_A = len(target_train[target_train == 0])
    N_H = len(target_train[target_train == 1])
    r = np.math.ceil((N_A - N_H) / len(aux_syn_tracks_unique_list))
    idx_human = np.bool_(np.zeros(len(aux_syn_data[0])))
    for tid_syn in aux_syn_tracks_unique_list:
        track_ind_total = np.where(aux_syn_tracks == tid_syn)[0]
        track_indices = np.random.choice(track_ind_total, size=min(len(track_ind_total), r), replace=False).tolist()
        idx_human[track_indices] = True

    # concatenate idx_human segments from synthetic set
    X = np.concatenate((X, aux_syn_data[0][idx_human]), axis=0)
    for key in train_data[1].keys():
        aux_syn_arr = np.array(aux_syn_data[1][key])
        train_data[1][key].extend(aux_syn_arr[idx_human].tolist())

    labels = train_data[1]
    if config.with_pca_augmentation:
        idx_syn = idx_human | idx_animal
        segid_arr = np.array(aux_syn_data[1]['segment_id'])
        required_indices = np.where(idx_syn == True)[0].tolist()
        required_segments = segid_arr[required_indices].tolist()
        syn_pca_data = data_parser.dump_PCA_synthetic_data(
            pickle_name='MAFAT RADAR Challenge - Auxiliary Synthetic Set V2.pkl',
            csv_name='MAFAT RADAR Challenge - Auxiliary Synthetic Set V2.csv', required_segments=required_segments, expansion_method='augmentation')
        X = np.concatenate((X, syn_pca_data[0]), axis=0)
        for key in train_data[1].keys():
            train_data[1][key].extend(syn_pca_data[1][key])
        labels = train_data[1]

    if config.with_pca_compression:
        idx_syn = idx_human | idx_animal
        segid_arr = np.array(aux_syn_data[1]['segment_id'])
        required_indices = np.where(idx_syn == True)[0].tolist()
        required_segments = segid_arr[required_indices].tolist()
        syn_pca_data = data_parser.dump_PCA_synthetic_data(
            pickle_name='MAFAT RADAR Challenge - Auxiliary Synthetic Set V2.pkl',
            csv_name='MAFAT RADAR Challenge - Auxiliary Synthetic Set V2.csv', required_segments=required_segments, expansion_method='compression')
        X = np.concatenate((X, syn_pca_data[0]), axis=0)
        for key in train_data[1].keys():
            train_data[1][key].extend(syn_pca_data[1][key])
        labels = train_data[1]

    if config.with_phase_time_shift:
        idx_syn = idx_human | idx_animal
        segid_arr = np.array(aux_syn_data[1]['segment_id'])
        required_indices = np.where(idx_syn == True)[0].tolist()
        required_segments = segid_arr[required_indices].tolist()
        syn_pca_data = data_parser.dump_phase_time_shift_synthetic_data(
            pickle_name='MAFAT RADAR Challenge - Auxiliary Synthetic Set V2.pkl',
            csv_name='MAFAT RADAR Challenge - Auxiliary Synthetic Set V2.csv', required_segments=required_segments)
        X = np.concatenate((X, syn_pca_data[0]), axis=0)
        for key in train_data[1].keys():
            train_data[1][key].extend(syn_pca_data[1][key])
        labels = train_data[1]

    if config.with_freq_rotation:
        idx_syn = idx_human | idx_animal
        segid_arr = np.array(aux_syn_data[1]['segment_id'])
        required_indices = np.where(idx_syn == True)[0].tolist()
        required_segments = segid_arr[required_indices].tolist()
        syn_freq_rotated_data = data_parser.dump_freq_rotation_synthetic_data(
            pickle_name='MAFAT RADAR Challenge - Auxiliary Synthetic Set V2.pkl',
            csv_name='MAFAT RADAR Challenge - Auxiliary Synthetic Set V2.csv', required_segments=required_segments)
        X = np.concatenate((X, syn_freq_rotated_data[0]), axis=0)
        for key in train_data[1].keys():
            train_data[1][key].extend(syn_freq_rotated_data[1][key])
        labels = train_data[1]

    if config.with_window_augmentation:
        idx_syn = idx_human | idx_animal
        segid_arr = np.array(aux_syn_data[1]['segment_id'])
        required_indices = np.where(idx_syn == True)[0].tolist()
        required_segments = segid_arr[required_indices].tolist()

        if len(config.window_list) == 0:
            raise Exception('Can\'t use window augmentation without window list')

        for window in config.window_list:
            syn_window_data = data_parser.dump_window_synthetic_data(window_type=window,
                                                                     pickle_name='MAFAT RADAR Challenge - Auxiliary Synthetic Set V2.pkl',
                                                                     csv_name='MAFAT RADAR Challenge - Auxiliary Synthetic Set V2.csv',
                                                                     required_segments=required_segments)
            X = np.concatenate((X, syn_window_data[0]), axis=0)
            for key in train_data[1].keys():
                train_data[1][key].extend(syn_window_data[1][key])
            labels = train_data[1]

    return X, labels


def truncate_data_to_even(train_data):
    target_train = np.array(train_data[1]['target_type'])
    N_A = len(target_train[target_train == 0])
    N_H = len(target_train[target_train == 1])
    traget_diff = N_A - N_H if N_A > N_H else N_H - N_A
    larger_target = 0 if N_A > N_H else 1
    # find idx to keep
    idx = len(train_data[0]) - 1
    idx_to_keep = np.bool_(np.ones(len(train_data[0])))
    while idx >= 0 and traget_diff > 0:
        if target_train[idx] == larger_target:
            idx_to_keep[idx] = False
            traget_diff -= 1
        idx -= 1
    # keep idx_to_keep
    X = train_data[0][idx_to_keep]
    labels = collections.OrderedDict()
    for key in train_data[1].keys():
        labels[key] = (np.array(train_data[1][key])[idx_to_keep]).tolist()

    return X, labels


def PCA_dimension_reducation(data, data_parser):
    X = data.reshape(data.shape[0], -1)
    new_X = np.array(data_parser.PCA_time_augmentation(X, method='compression'))
    Y = new_X.reshape(data.shape)
    return Y


def load_data(config):
    if config.load_pkl_data:
        train_pkl = config.train_pkl_file
        validation_pkl = config.validation_pkl_file
        if (os.path.exists(train_pkl) and os.path.exists(validation_pkl)) is False:
            raise Exception('did not found ONE of the following files:\ntrain_pkl:{}\nvalidation_pkl:{}'.format(train_pkl, validation_pkl))
        train_data_from_pkl = pickle.load(open(train_pkl, "rb"))
        train_data = np.array([(iq_mat, label) for iq_mat, label in zip(train_data_from_pkl[0], train_data_from_pkl[1])])
        validation_data_from_pkl = pickle.load(open(validation_pkl, "rb"))
        validation_data = np.array([(iq_mat, label) for iq_mat, label in zip(validation_data_from_pkl[0], validation_data_from_pkl[1])])
        # # validation in debug with config.load_pkl_data == False
        # False in [(x == x2).all() for x, x2 in zip(train_data[:, 0], train_data2[:, 0])]
        # False in [x == x2 for x, x2 in zip(train_data[:, 1], train_data2[:, 1])]

        data = convert_numpy_to_dataset(train_data, validation_data, config)
        transformed_data = transform(data)

        data_iterators = make_iterators(transformed_data, config)
    else:
        train_data, validation_data, data_parser = read_data(config)

        assert not (config.load_low_human_experiment and config.stable_mode)
        assert not (config.load_synthetic and config.stable_mode)
        assert not (config.load_synthetic and (config.snr_type == 'high'))
        # assert not (config.load_synthetic and config.load_all_experiment)

        if config.augment_by_track is True:  # augment_by_track the human examples
            if config.tcn_use_variable_length and bool(re.search('tcn', config.exp_name, re.IGNORECASE)):
                raise Exception('variable length should not be used with augment by track')
            train_data = expand_human_data_by_tracks(train_data=train_data, config=config)

        if config.drop_geolocation is True:
            train_data = drop_geolocation(train_data=train_data, config=config)

        if config.with_pca_augmentation is True:
            train_data = expand_data_by_pca_time_augmentation(data_parser=data_parser, train_data=train_data,
                                                              validation_segments_id_list=validation_data[1]['segment_id'])
        if config.with_pca_compression is True:
            train_data = expand_data_by_pca_time_comperssion(data_parser=data_parser, train_data=train_data,
                                                             validation_segments_id_list=validation_data[1]['segment_id'])
        if config.with_phase_time_shift is True:
            train_data = expand_data_by_phase_time_shift(data_parser=data_parser, train_data=train_data,
                                                         validation_segments_id_list=validation_data[1]['segment_id'])
        if config.with_freq_rotation is True:
            train_data = expand_data_by_freq_rotation(data_parser=data_parser, train_data=train_data,
                                                      validation_segments_id_list=validation_data[1]['segment_id'])
        if config.with_window_augmentation is True:
            train_data = expand_data_by_window_function(data_parser=data_parser, train_data=train_data,
                                                        validation_segments_id_list=validation_data[1]['segment_id'], config=config)
        if config.stable_mode is True:
            train_data = expand_human_data_by_aux_set(data_parser=data_parser, train_data=train_data)

        if config.load_low_human_experiment is True:
            train_data = expand_data_by_human_low_exp_set(data_parser=data_parser, train_data=train_data, config=config)

        if config.load_synthetic is True:
            train_data = expand_data_by_synthetic_set(data_parser=data_parser, train_data=train_data, config=config, validation_data=validation_data)

        if config.with_rect_augmentation is True:
            train_data = expand_data_by_sampling_rect(data=train_data, config=config)
            validation_data = expand_data_by_sampling_rect(data=validation_data, config=config)

        if config.tcn_use_variable_length and bool(re.search('tcn', config.exp_name, re.IGNORECASE)):
            train_data = resahpe_data_to_var_length(train_data=train_data)

        if config.truncate_data_to_even:
            train_data = truncate_data_to_even(train_data=train_data)

        if config.with_pca_dimension_reducation is True:
            Y = PCA_dimension_reducation(train_data[0], data_parser)
            train_data = Y,train_data[1]

        train_data, validation_data = convert_metadata_to_numpy(train_data, validation_data)

        NUMBER_OF_SEGEMENTS_BEFORE_AUGMENTION = len(train_data)

        if config.augment_per_epoch is False:
            train_data = augment(train_data, config)

        # swap axes for sequential Data
        # if config.exp_name == "LSTM":
        #     data_iterators = convert_data_to_sequential(data_iterators)
        if bool(re.search('lstm', config.exp_name, re.IGNORECASE)) is True:
            train_data, validation_data = lstm_preprocess_data(train_data, validation_data, config)

        if bool(re.search('tcn', config.exp_name, re.IGNORECASE)) is True:
            train_data, validation_data = tcn_preprocess_data(train_data, validation_data, config)

        """ Background Data"""
        if config.learn_background:
            train_data = append_background_data_to_train(train_data, data_parser, config.background_num)
            validation_data = np.array([(iq_mat, to_categorical(label, num_classes=2, dtype='float32')) for iq_mat, label in
                                        zip(validation_data[:, 0], validation_data[:, 1])])
            train_data = np.array([(iq_mat, to_categorical(label, num_classes=3, dtype='float32')) for iq_mat, label in
                                   zip(train_data[:, 0], train_data[:, 1])])

            # train_data = reshape_label(train_data)
            # validation_data = reshape_label(validation_data)

        print_data_distribution_before_transform_to_dataset(train_data, validation_data)

        if config.save_pkl_data is True:
            if os.path.exists('Organized_Data_in_pickle') is False:
                if 'dataset' not in os.getcwd():
                    raise Exception('Trying to save pickle file outside of dataset directory!')
                os.mkdir('Organized_Data_in_pickle')

            pickle.dump([train_data[:, 0], train_data[:, 1]],
                        open("Organized_Data_in_pickle/{}_train_{}_segments.pkl".format(config.pickle_file_name, len(train_data)), "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump([validation_data[:, 0], validation_data[:, 1]],
                        open("Organized_Data_in_pickle/{}_validation_{}_segments.pkl".format(config.pickle_file_name, len(validation_data)), "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)
            print("dumped train data to file: Organized_Data_in_pickle/{}_train_{}_segments.pkl".format(config.pickle_file_name, len(train_data)))
            print("dumped validation data to file: Organized_Data_in_pickle/{}_validation_{}_segments.pkl".format(config.pickle_file_name,
                                                                                                                  len(validation_data)))
            with open('Organized_Data_in_pickle/{}_data_description.txt'.format(config.pickle_file_name), mode='w') as data_file:

                data_file.write('No. segments in train: {}'.format(len(train_data)))
                data_file.write('No. segments in validation: {}'.format(len(validation_data)))
                data_file.write('train targets: {} humans, {} animals'.format(len(train_data[:, 1][train_data[:, 1] == 1]),
                                                                              len(train_data[:, 1][train_data[:, 1] == 0])))
                data_file.write('validation targets: {} humans, {} animals'.format(len(validation_data[:, 1][validation_data[:, 1] == 1]),
                                                                                   len(validation_data[:, 1][validation_data[:, 1] == 0])))

                data_file.write('No. segments in train: {}'.format(len(train_data)))
                data_file.write('No. segments in validation: {}'.format(len(validation_data)))

                for key in config.keys():
                    data_file.write('config[{}] = {}'.format(key, config[key]))

        data = convert_numpy_to_dataset(train_data, validation_data, config)

        # calculate number of elements in train data after augmentation funcs
        count = NUMBER_OF_SEGEMENTS_BEFORE_AUGMENTION # for original data
        count += 3 * NUMBER_OF_SEGEMENTS_BEFORE_AUGMENTION if 'timestep_shift' in config.augment_funcs else 0
        count += 3 * NUMBER_OF_SEGEMENTS_BEFORE_AUGMENTION if 'normal' in config.augment_funcs else 0
        count += 3 * NUMBER_OF_SEGEMENTS_BEFORE_AUGMENTION if 'row_shift' in config.augment_funcs else 0
        count += 1 * NUMBER_OF_SEGEMENTS_BEFORE_AUGMENTION if 'horiz_flip' in config.augment_funcs else 0
        count += 1 * NUMBER_OF_SEGEMENTS_BEFORE_AUGMENTION if 'flip_image' in config.augment_funcs else 0
        NUMBER_OF_SEGEMENTS_AFTER_AUGMENTION = count

        if config.steps_per_epoch_overwrite is False:
            config.__setattr__("steps_per_epoch", NUMBER_OF_SEGEMENTS_AFTER_AUGMENTION // config.batch_size)

        transformed_data = transform(data)

        if config.augment_per_epoch is True:
            transformed_data = transform_per_epoch(transformed_data, config)
            print(20 * '#')
            print('Data COUNT AFTER conversion to tf.data.Dataset')
            print('No. segments in train: {}'.format(sum(1 for _ in transformed_data['train'].as_numpy_iterator())))
            print('No. segments in validation: {}'.format(sum(1 for _ in transformed_data['train_eval'].as_numpy_iterator())))


        data_iterators = make_iterators(transformed_data, config)

    return data_iterators


def read_data(config):
    stable_mode = config.get('stable_mode')
    testset_size = config.get('N_test')
    data_parser = DataSetParser(stable_mode=stable_mode, config=config)

    train_data = data_parser.get_dataset_by_snr(dataset_type='train', snr_type=config.snr_type)

    # TODO: sync with stefan after he finishes the validation over low and high

    if config.use_public_test_set:
        # use MAFAT public test set as validation
        validation_data = data_parser.get_dataset_by_snr(dataset_type='validation', snr_type=config.snr_type)
    else:
        # sort train_data by track id to avoid over fitting over the validation data
        train_data, validation_data = generate_test_set(train_data, testset_size, config.snr_type)
        # noise half of validation_data where target == 1 (human)
        if config.snr_type == 'all':
            validation_data = noise_human_validation_data(validation_data)

    # validate tracks
    validate_train_val_tracks(train_tracks=train_data[1]['track_id'], valid_tracks=validation_data[1]['track_id'])

    return train_data, validation_data, data_parser


def transform_per_epoch(data, config):
    def augment_normal(iq_mat, label):
        iq_mat = iq_mat + tf.random.normal(shape=np.array(iq_mat.shape), mean=config.augment_normal_mean, stddev=config.augment_normal_std,
                                           dtype=tf.float32)
        return iq_mat, label

    def augment_timestep_shift(iq_mat, label):
        max_shift = tf.random.uniform(shape=[], minval=1 , maxval=config.timestep_shift_delta, dtype=tf.int32)
        shift = tf.random.uniform(shape=[], minval=-max_shift, maxval=max_shift + 1, dtype=tf.int32)
        # shift = np.random.choice(
        #     [x for x in range(-config.timestep_shift_delta, config.timestep_shift_delta + 1) if x != 0])
        return tf.roll(iq_mat, shift, axis=1), label

    def augment_vertical_flip(iq_mat, label):
        return tf.image.flip_up_down(iq_mat), label

    def augment_horiz_flip(iq_mat, label):
        return tf.image.flip_left_right(iq_mat), label

    # orig_data = data['train']
    datasets_list = []
    for augment_index in range(config.augment_expansion_number):
        if 'flip_image' in config.augment_funcs and augment_index == 0:
            datasets_list.append(data['train'].map(augment_vertical_flip))
        if 'horiz_flip' in config.augment_funcs and augment_index == 0:
            datasets_list.append(data['train'].map(augment_horiz_flip))
        if 'normal' in config.augment_funcs:
            datasets_list.append(data['train'].map(augment_normal))
        if 'timestep_shift' in config.augment_funcs:
            datasets_list.append(data['train'].map(augment_timestep_shift))

    for dataset in datasets_list:
        data['train'] = data['train'].concatenate(dataset)


    # vertical_data = data['train'].map(augment_vertical_flip)
    # horiz_data = data['train'].map(augment_horiz_flip)
    # noise_data = data['train'].map(augment_normal).repeat(config.augment_expansion_number)
    # step_shift_data = data['train'].map(augment_timestep_shift).repeat(config.augment_expansion_number)
    # data['train'] = orig_data.concatenate(noise_data).concatenate(step_shift_data).concatenate(horiz_data).concatenate(vertical_data)

    # n_train = 150000
    # data['train'] = data['train'].shuffle(n_train + 1000).repeat()

    return data


def transform(data):
    def transform_example(iq_mat, label):
        iq_mat, label = tf.cast(iq_mat, tf.float32), tf.cast(label, tf.int8)
        return iq_mat, label

    data['train'] = data['train'].map(transform_example).cache()
    data['train_eval'] = data['train_eval'].map(transform_example).cache()

    return data


def augment(data, config):
    def augment_normal(iq_mat, label):
        iq_mat = iq_mat + np.random.normal(loc=config.augment_normal_mean, scale=config.augment_normal_std,
                                           size=iq_mat.shape)
        return iq_mat, label

    def augment_gaussian_filt_2d(iq_mat, label):
        iq_mat = gaussian_filter(iq_mat, sigma=1)
        return iq_mat, label

    def augment_freq_shift(iq_mat, label):
        if np.random.binomial(1, 0.5, size=1)[0] == 0:
            shift = config.freq_shift_delta
        else:
            shift = -config.freq_shift_delta
        if config.shift_freq_dc_width == 0:
            return np.roll(iq_mat, shift, axis=0), label
        else:
            shiftwidth = config.shift_freq_dc_width
            iq_mat[shiftwidth:-shiftwidth, :] = np.roll(iq_mat[shiftwidth:-shiftwidth, :], shift, axis=0)
            return iq_mat, label

    def augment_timestep_shift(iq_mat, label):
        if np.random.binomial(1, 0.5, size=1)[0] == 0:
            shift = config.timestep_shift_delta
        else:
            shift = -config.timestep_shift_delta
        # shift = np.random.choice(
        #     [x for x in range(-config.timestep_shift_delta, config.timestep_shift_delta + 1) if x != 0])
        return np.roll(iq_mat, shift, axis=1), label

    def augment_vertical_flip(iq_mat, label):
        return np.flipud(iq_mat), label

    def augment_horiz_flip(iq_mat, label):
        return np.fliplr(iq_mat), label

    def get_data_to_augment(data_pipeline, orig_data_pipeline):
        if config.pipeline_data_augmentation:
            return data_pipeline
        else:
            return orig_data_pipeline

    def augment_with_pca(iq_mat, label):
        return PCA_expansion(iq_mat, config), label

    def row_shift(iq_mat, label):
        iq2 = np.zeros(iq_mat.shape)
        RESAMPLE_FACTOR = config.row_shift_resample_factor
        LIMIT = config.row_shift_limit
        for i, shift in zip(range(iq_mat.shape[0]), np.random.choice([x for x in range(-LIMIT, LIMIT + 1)], iq_mat.shape[0]).tolist()):
            # iq2[i,:] = np.roll(iq_mat[i,:], shift, axis=0) for pixel-wise resolution
            iq2[i, :] = resample_poly(np.roll(resample_poly(iq_mat[i, :], up=RESAMPLE_FACTOR, down=1), shift, axis=0), up=1, down=RESAMPLE_FACTOR)
        iq2 = np.maximum(np.median(iq_mat) - 1, iq2)
        iq2 = (iq2 - np.mean(iq2)) / np.std(iq2)
        # plt.figure()
        # plt.imshow(iq_mat)
        # plt.savefig('row_shift org iq')
        # plt.figure()
        # plt.imshow(iq2)
        # plt.savefig('row_shift iq')
        # print('cov distance: {}'.format(np.linalg.norm(np.cov(iq2,iq_mat))/ np.linalg.norm(np.cov(iq_mat))))
        return iq2, label

    orig_data = data
    augment_funcs = config.augment_funcs
    for augment_index in range(config.augment_expansion_number):
        if 'flip_image' in augment_funcs and augment_index == 0:
            data = np.concatenate(
                (data, np.array([augment_vertical_flip(sample[0], sample[1]) for sample in
                                 get_data_to_augment(data, orig_data)])), axis=0)
        if 'horiz_flip' in augment_funcs and augment_index == 0:
            data = np.concatenate((data, np.array(
                [augment_horiz_flip(sample[0], sample[1]) for sample in get_data_to_augment(data, orig_data)])),
                                  axis=0)
        if 'normal' in augment_funcs:
            data = np.concatenate((data, np.array(
                [augment_normal(sample[0], sample[1]) for sample in get_data_to_augment(data, orig_data)])), axis=0)
        if 'guassian_filt' in augment_funcs:
            data = np.concatenate(
                (data, np.array([augment_gaussian_filt_2d(sample[0], sample[1]) for sample in
                                 get_data_to_augment(data, orig_data)])), axis=0)
        if 'freq_shift' in augment_funcs:
            data = np.concatenate((data, np.array(
                [augment_freq_shift(sample[0], sample[1]) for sample in get_data_to_augment(data, orig_data)])),
                                  axis=0)
        if 'timestep_shift' in augment_funcs:
            data = np.concatenate(
                (data, np.array([augment_timestep_shift(sample[0], sample[1]) for sample in
                                 get_data_to_augment(data, orig_data)])), axis=0)

        if 'pca' in augment_funcs:
            data = np.concatenate((data, np.array(
                [augment_with_pca(sample[0], sample[1]) for sample in get_data_to_augment(data, orig_data)])), axis=0)

        if 'row_shift' in augment_funcs:
            data = np.concatenate((data, np.array(
                [row_shift(sample[0], sample[1]) for sample in get_data_to_augment(data, orig_data)])), axis=0)

    return data


def make_iterators(data, config):
    # train_iter = data['train'].map(augment_example).shuffle(1000).batch(config.batch_size, drop_remainder=True).take(-1)
    # n_train = data['train'].__len__().numpy()
    n_train = 150000
    if config.augment_per_epoch is True:
        train_iter = data['train'].shuffle(n_train + 1000).repeat().batch(config.batch_size, drop_remainder=True).prefetch(config.batch_size).take(-1)
    else:
        train_iter = data['train'].shuffle(n_train + 1000).batch(config.batch_size, drop_remainder=True).take(-1)
    train_eval_iter = data['train_eval'].batch(config.batch_size_eval).take(-1)

    iterators = {'train': train_iter,
                 'train_eval': train_eval_iter}
    return iterators


def drop_geolocation(train_data, config):
    indices = [i for i, x in enumerate(train_data[1]['geolocation_id']) if x == '3']
    mask = np.ones(len(train_data[0]), dtype=bool)
    mask[indices] = False
    X = train_data[0][mask]
    labels = copy.deepcopy(train_data[1])
    # convert to numpy
    for key in labels.keys():
        labels[key] = np.array(labels[key])
        labels[key] = labels[key][mask]
        labels[key] = labels[key].tolist()

    return X, labels


def resahpe_data_to_var_length(train_data):
    # sort train_data by track_id
    X_sorted, labels_sorted = sort_data_by_label(train_data, label='track_id', validate=True)
    # use np listing
    track_list_np = np.array([int(tid) for tid in labels_sorted['track_id']])
    track_list_unique_np = np.unique(track_list_np)
    # new train will be (X_sorted_reshaped, labels_sorted_reshaped)
    # X_sorted_reshaped = np.zeros(shape=[track_list_unique_np.shape[0], X_sorted.shape[1], X_sorted.shape[2]])
    X_sorted_reshaped = []
    labels_sorted_reshaped = collections.OrderedDict()
    for key in labels_sorted.keys():
        labels_sorted_reshaped[key] = []
    # concatenate each track to a single variable length segment
    for tid in track_list_unique_np:
        indices = np.where(track_list_np == tid)[0]
        X_sorted_reshaped.append(np.concatenate(X_sorted[indices], axis=-1))
        for key in labels_sorted.keys():
            labels_sorted_reshaped[key].append(labels_sorted[key][indices[0]])

    return X_sorted_reshaped, labels_sorted_reshaped


def append_background_data_to_train(train_data, data_parser, background_num):
    # WorkAround for Sweeps:
    if background_num == 0:
        return train_data
    # convert 'empty' label to number
    data_parser.aux_background_data[1]['target_type'] = [2 if y == 'empty' else 9 for y in
                                                         data_parser.aux_background_data[1]['target_type']]
    assert 9 not in data_parser.aux_background_data[1]['target_type']
    # concatenate
    aux_data = np.array([(iq_mat, label) for iq_mat, label in
                         zip(data_parser.aux_background_data[0][:background_num],
                             data_parser.aux_background_data[1]['target_type'][:background_num])])
    train_data = np.concatenate((train_data, aux_data), axis=0)

    return train_data


def print_target_distribution(train_data, validation_data):
    def count_labels(data):
        human_label = 1
        n_human = 0
        animal_label = 0
        n_animal = 0
        empty_label = 2
        n_empty = 0
        train_targets = data[:, 1]
        for t in train_targets:
            if np.equal(t, human_label).all():
                n_human = n_human + 1
            elif np.equal(t, animal_label).all():
                n_animal = n_animal + 1
            elif np.equal(t, empty_label).all():
                n_empty = n_empty + 1
            else:
                raise Exception('Unexpected label, not human/animal/background')
        assert len(data) == n_human + n_animal + n_empty
        return n_human, n_animal, n_empty

    n_human_train, n_animal_train, n_empty_train = count_labels(train_data)
    n_human_valid, n_animal_valid, n_empty_valid = count_labels(validation_data)

    print(30 * '#')
    print('train_data label distribution:')
    print('n_human = {}, n_animal = {}, n_empty = {}'.format(n_human_train, n_animal_train, n_empty_train))
    print('validation_data label distribution:')
    print('n_human = {}, n_animal = {}, n_empty = {}'.format(n_human_valid, n_animal_valid, n_empty_valid))
    print(30 * '#')
