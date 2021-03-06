import copy
import tensorflow as tf
import tensorflow_datasets as tfds
from data.data_parser import *
from tensorflow.keras.utils import to_categorical
from data.data_loader_utils import *
from scipy.ndimage import gaussian_filter
from data.signal_processing import *
import re


def generator(data, config):
    batch_features = np.zeros((config.batch_size, config.rect_augment_num_of_rows, config.rect_augment_num_of_cols, 1))
    batch_labels = np.zeros((config.batch_size, 1))

    while True:
        for batch_X, batch_y in data:
            X, y = expand_data_generator_by_sampling_rect(batch_X, batch_y, config)
            array_indices = np.arange(len(X))
            for batch_repeat_index in range(config.repeat_rect_exp_per_batch):
                for i in range(config.batch_size):
                    # choose random index in features
                    index = np.random.choice(array_indices, 1)[0]
                    # Delete index from the future options
                    array_indices = np.delete(array_indices, np.where(index == array_indices)[0][0])
                    # Crop the image
                    batch_features[i] = X[index]
                    batch_labels[i] = y[index]
            yield (batch_features, batch_labels)


def generate_test_set(train_data, TRACK_ID_TEST_SET_SIZE):
    """
    Since test date given is un-labeled, in order to evaluate the the model properly we need to hold-out training
    examples from the train set to be used as test.
    The train set is divided into tracks so we need to make sure there are no segments from same tracks in train & test
    to avoid inherited over-fitting.
    """
    # save for validation after split
    train_data_orig = train_data

    # sort train_data by snr_type
    # X_sorted_snr, labels_sorted_snr = sort_data_by_label(train_data, label='snr_type')
    # split to 'HighSNR'/'LowSNR'
    high_data, high_indices = get_data_by_snr(snr_type='HighSNR', X=train_data_orig[0], labels=train_data_orig[1])
    low_data, low_indices = get_data_by_snr(snr_type='LowSNR', X=train_data_orig[0], labels=train_data_orig[1])
    assert (not ('LowSNR' in high_data[1]['snr_type'])) or (not ('HighSNR' in low_data[1]['snr_type']))
    assert not (True in (low_indices & high_indices))

    # mixed indices - of tracks with both high and low snr
    # high_indices, low_indices BOTH are arrays that have no duality snr
    mix_idx = np.bool_(np.ones(len(high_indices))) & np.logical_not(high_indices | low_indices)
    X_train_mixed = train_data_orig[0][mix_idx]
    labels_train_mixed = collections.OrderedDict()
    for key in train_data_orig[1].keys():
        labels_train_mixed[key] = (np.array(train_data_orig[1][key])[mix_idx]).tolist()

    # sort high_data and low_data by track_id
    X_sorted_high, labels_sorted_high = sort_data_by_track_id(high_data)
    X_sorted_low, labels_sorted_low = sort_data_by_track_id(low_data)

    # split to human/animal data for HighSNR
    human_data_high = get_data_by_target(target=1, X_sorted=X_sorted_high, labels_sorted=labels_sorted_high)
    animal_data_high = get_data_by_target(target=0, X_sorted=X_sorted_high, labels_sorted=labels_sorted_high)
    assert (not (0 in human_data_high[1]['target_type'])) or (not (1 in animal_data_high[1]['target_type']))
    # imax is the last index (INCLUDING!) of the validation data
    imax_valid_animal_high = get_i_to_split(target_data=animal_data_high, counter=TRACK_ID_TEST_SET_SIZE / 4)
    imax_valid_human_high = get_i_to_split(target_data=human_data_high, counter=TRACK_ID_TEST_SET_SIZE / 2)

    # split to human/animal data for LowSNR
    human_data_low = get_data_by_target(target=1, X_sorted=X_sorted_low, labels_sorted=labels_sorted_low)
    animal_data_low = get_data_by_target(target=0, X_sorted=X_sorted_low, labels_sorted=labels_sorted_low)
    assert (not (0 in human_data_low[1]['target_type'])) or (not (1 in animal_data_low[1]['target_type']))
    # imax is the last index (INCLUDING!) of the validation data
    imax_valid_animal_low = get_i_to_split(target_data=animal_data_low, counter=TRACK_ID_TEST_SET_SIZE / 4)
    # NO human data with low snr that does not have a duplicated snr --> human data in validation is only from high snr
    # imax_valid_human_low = get_i_to_split(target_data=human_data_low, counter=TRACK_ID_TEST_SET_SIZE / 4)

    # human data high
    X_valid_human_high, labels_valid_human_high, X_train_human_high, labels_train_human_high = split_train_and_valid_by_target(
        target_data=human_data_high, imax=imax_valid_human_high)
    # animal data high
    X_valid_animal_high, labels_valid_animal_high, X_train_animal_high, labels_train_animal_high = split_train_and_valid_by_target(
        target_data=animal_data_high, imax=imax_valid_animal_high)
    # human data low
    # X_valid_human_low, labels_valid_human_low, X_train_human_low, labels_train_human_low = split_train_and_valid_by_target(
    #     target_data=human_data_low, imax=imax_valid_human_low)
    # animal data low
    X_valid_animal_low, labels_valid_animal_low, X_train_animal_low, labels_train_animal_low = split_train_and_valid_by_target(
        target_data=animal_data_low, imax=imax_valid_animal_low)

    # split train and validation total
    X_valid = np.concatenate((X_valid_human_high, X_valid_animal_high, X_valid_animal_low), axis=0)
    X_train = np.concatenate((X_train_human_high, X_train_animal_high, X_train_animal_low, X_train_mixed), axis=0)
    labels_train = collections.OrderedDict()
    labels_valid = collections.OrderedDict()
    for key in train_data_orig[1].keys():
        # validation
        labels_valid[key] = labels_valid_human_high[key]
        labels_valid[key].extend(labels_valid_animal_high[key])
        labels_valid[key].extend(labels_valid_animal_low[key])
        # train
        labels_train[key] = labels_train_human_high[key]
        labels_train[key].extend(labels_train_animal_high[key])
        labels_train[key].extend(labels_train_animal_low[key])
        labels_train[key].extend(labels_train_mixed[key])

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

    train_data = (X_train, labels_train)
    validation_data = (X_valid, labels_valid)
    assert len(train_data[0]) + len(validation_data[0]) == len(train_data_orig[0])
    return train_data, validation_data


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


def expand_data_by_time_phase_distortion(data_parser, train_data, validation_segments_id_list):
    # Validate that the interpolated data set has no bias with the validation set
    augmentad_data = data_parser.dump_time_phase_distortion_data(pickle_name='MAFAT RADAR Challenge - Training Set V1.pkl',
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


def expand_data_by_interpolation_fast_axis(data_parser, train_data, validation_segments_id_list):
    interpolated_data = data_parser.get_interpolation_data_fast_axis(
        validation_segments_id_list=validation_segments_id_list)

    # Validate that the interpolated data set has no bias with the validation set
    for segment_id in interpolated_data[1]['segment_id']:
        if segment_id[1:] in validation_segments_id_list:
            raise Exception('Using interpolation mode and data is biased with the validation data set !!')
    X = np.concatenate((train_data[0], interpolated_data[0]), axis=0)  # Stack the interpolated data
    train_labels = train_data[1]
    train_interpolated_labels = interpolated_data[1]
    for key in train_labels.keys():
        train_labels[key].extend(train_interpolated_labels[key])
    labels = train_labels
    # # validate 50/50 distribution
    # num_of_animals = sum([1 for target in train_data[1]['target_type'] if target == 0])
    # num_of_humans = sum([1 for target in train_data[1]['target_type'] if target == 1])
    # if abs(num_of_animals / (num_of_animals + num_of_humans) - 0.5) > 0.05:
    #     raise Exception('Using stable mode and data resulted as unstable !!')
    return X, labels


def expand_data_by_interpolation_slow_axis(data_parser, train_data, validation_segments_id_list):
    interpolated_data = data_parser.get_interpolation_data_slow_axis(
        validation_segments_id_list=validation_segments_id_list)

    # Validate that the interpolated data set has no bias with the validation set
    for segment_id in interpolated_data[1]['segment_id']:
        if segment_id[1:] in validation_segments_id_list:
            raise Exception('Using interpolation mode and data is biased with the validation data set !!')
    X = np.concatenate((train_data[0], interpolated_data[0]), axis=0)  # Stack the interpolated data
    train_labels = train_data[1]
    train_interpolated_labels = interpolated_data[1]
    for key in train_labels.keys():
        train_labels[key].extend(train_interpolated_labels[key])
    labels = train_labels
    # # validate 50/50 distribution
    # num_of_animals = sum([1 for target in train_data[1]['target_type'] if target == 0])
    # num_of_humans = sum([1 for target in train_data[1]['target_type'] if target == 1])
    # if abs(num_of_animals / (num_of_animals + num_of_humans) - 0.5) > 0.05:
    #     raise Exception('Using stable mode and data resulted as unstable !!')
    return X, labels


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
        if len(expanded_iq) == 0:
            print('While trying to sample index {} didn\'t managed to sample anything'.format(index))
            continue
        sampled_indices = np.random.choice(len(expanded_iq),
                                           size=min(config.sample_for_segment_rect_augmentation, len(expanded_iq)),
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


def expand_data_by_human_low_exp_set(data_parser, train_data, with_pca):
    aux_exp_data = data_parser.get_dataset_allsnr(dataset_type='aux_exp')
    aux_exp_snr_type = np.array(aux_exp_data[1]['snr_type'])
    idx_low_list = np.where(aux_exp_snr_type == 'LowSNR')[0].tolist()
    idx_low = np.bool_(np.zeros(len(aux_exp_data[0])))
    idx_low[idx_low_list] = True

    X = np.concatenate((train_data[0], aux_exp_data[0][idx_low]), axis=0)  # Stack the experiment data
    train_labels = train_data[1]
    for key in train_labels.keys():
        train_labels[key].extend(np.array(aux_exp_data[1][key])[idx_low].tolist())
    labels = train_labels
    if with_pca:
        required_indices = np.where(idx_low == True)[0].tolist()
        segid_arr = np.array(aux_exp_data[1]['segment_id'])
        required_segments = segid_arr[required_indices].tolist()
        exp_pca_data = data_parser.dump_PCA_synthetic_data(
            pickle_name='MAFAT RADAR Challenge - Auxiliary Experiment Set V2.pkl',
            csv_name='MAFAT RADAR Challenge - Auxiliary Experiment Set V2.csv', required_segments=required_segments)
        X = np.concatenate((X, exp_pca_data[0]), axis=0)
        for key in train_data[1].keys():
            train_data[1][key].extend(exp_pca_data[1][key])
        labels = train_data[1]

    return X, labels


def expand_data_by_synthetic_set(data_parser, train_data, with_pca):
    aux_syn_data = data_parser.get_dataset_allsnr(dataset_type='aux_syn')
    aux_syn_tracks = np.array(aux_syn_data[1]['track_id'])
    train_tracks = np.array(train_data[1]['track_id'])
    aux_syn_target = np.array(aux_syn_data[1]['target_type'])

    # concatenate all animal segments from synthetic set since it is with mostly human
    # idx_animal_debug = np.where(aux_syn_target == 0)[0].tolist()
    idx_animal_list = np.where(aux_syn_target == 0)[0].tolist()
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
    if with_pca:
        idx_syn = idx_human | idx_animal
        segid_arr = np.array(aux_syn_data[1]['segment_id'])
        required_indices = np.where(idx_syn == True)[0].tolist()
        required_segments = segid_arr[required_indices].tolist()
        syn_pca_data = data_parser.dump_PCA_synthetic_data(
            pickle_name='MAFAT RADAR Challenge - Auxiliary Synthetic Set V2.pkl',
            csv_name='MAFAT RADAR Challenge - Auxiliary Synthetic Set V2.csv', required_segments=required_segments)
        X = np.concatenate((X, syn_pca_data[0]), axis=0)
        for key in train_data[1].keys():
            train_data[1][key].extend(syn_pca_data[1][key])
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


def load_data(config):
    train_data, validation_data, data_parser = read_data(config)
    # train_data = data_parser.get_dataset_by_snr(dataset_type='train',snr_type='HighSNR')

    assert not (config.load_low_human_experiment and config.stable_mode)
    assert not (config.load_synthetic and config.stable_mode)
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
    if config.with_phase_distortion is True:
        train_data = expand_data_by_time_phase_distortion(data_parser=data_parser, train_data=train_data,
                                                          validation_segments_id_list=validation_data[1]['segment_id'])
    # if config.with_interpolation_fast_axis is True:
    #     train_data = expand_data_by_interpolation_fast_axis(data_parser=data_parser, train_data=train_data,
    #                                                         validation_segments_id_list=validation_data[1][
    #                                                             'segment_id'])
    #
    # if config.with_interpolation_slow_axis is True:
    #     train_data = expand_data_by_interpolation_slow_axis(data_parser=data_parser, train_data=train_data,
    #                                                         validation_segments_id_list=validation_data[1][
    #                                                             'segment_id'])
    if config.stable_mode is True:
        train_data = expand_human_data_by_aux_set(data_parser=data_parser, train_data=train_data)

    if config.load_low_human_experiment is True:
        train_data = expand_data_by_human_low_exp_set(data_parser=data_parser, train_data=train_data,
                                                      with_pca=config.with_pca_augmentation)
    if config.load_synthetic is True:
        train_data = expand_data_by_synthetic_set(data_parser=data_parser, train_data=train_data,
                                                  with_pca=config.with_pca_augmentation)
    if config.with_rect_augmentation is True:
        validation_data = expand_data_by_sampling_rect(data=validation_data, config=config)

    if config.with_preprocess_rect_augmentation is True:
        train_data = expand_data_by_sampling_rect(data=train_data, config=config)
        validation_data = expand_data_by_sampling_rect(data=validation_data, config=config)

    assert config.with_preprocess_rect_augmentation != config.with_rect_augmentation or (
            not config.with_preprocess_rect_augmentation and not config.with_rect_augmentation)

    if config.tcn_use_variable_length and bool(re.search('tcn', config.exp_name, re.IGNORECASE)):
        train_data = resahpe_data_to_var_length(train_data=train_data)

    if config.truncate_data_to_even:
        train_data = truncate_data_to_even(train_data=train_data)

    '''
    Clean signal processing algorithms
    '''

    train_data, validation_data = convert_metadata_to_numpy(train_data, validation_data)

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

    """ label distribution to print in logs """
    # print_target_distribution(train_data, validation_data)
    print(20 * '#')
    print('No. segments in train: {}'.format(len(train_data)))
    print('No. segments in validation: {}'.format(len(validation_data)))
    print('train targets: {} humans, {} animals'.format(len(train_data[:, 1][train_data[:, 1] == 1]),
                                                        len(train_data[:, 1][train_data[:, 1] == 0])))
    print('validation targets: {} humans, {} animals'.format(len(validation_data[:, 1][validation_data[:, 1] == 1]),
                                                             len(validation_data[:, 1][validation_data[:, 1] == 0])))

    print(20 * '#')
    print('No. segments in train: {}'.format(len(train_data)))
    print('No. segments in validation: {}'.format(len(validation_data)))

    data = convert_numpy_to_dataset(train_data, validation_data, config)

    transformed_data = transform(data)

    data_iterators = make_iterators(transformed_data, config)

    return data_iterators


def read_data(config):
    stable_mode = config.get('stable_mode')
    testset_size = config.get('N_test')
    data_parser = DataSetParser(stable_mode=stable_mode, config=config)

    # data_parser.plot_single_timestamp_frequency()
    # data_parser.plot_IQ_data()

    if config.snr_type == 'all':
        train_data = data_parser.get_dataset_allsnr(dataset_type='train')
    elif config.snr_type == 'low':
        train_data = data_parser.get_dataset_by_snr(dataset_type='train', snr_type='LowSNR')
    elif config.snr_type == 'high':
        train_data = data_parser.get_dataset_by_snr(dataset_type='train', snr_type='HighSNR')
    else:
        raise Exception('Unvalid SNR type please enter a valid one')

    # TODO: fix validation set for low and high snr
    if config.snr_type == 'low' or config.snr_type == 'high':
        raise Exception('need to validate low/high snr')
    # sort train_data by track id to avoid over fitting over the validation data
    train_data, validation_data = generate_test_set(train_data, testset_size)

    # noise half of validation_data where target == 1 (human)
    validation_data = noise_human_validation_data(validation_data)

    # validate tracks
    validate_train_val_tracks(train_tracks=train_data[1]['track_id'], valid_tracks=validation_data[1]['track_id'])

    return train_data, validation_data, data_parser


def transform(data):
    def transform_example(iq_mat, label):
        iq_mat, label = tf.cast(iq_mat, tf.float32), tf.cast(label, tf.int8)
        return iq_mat, label

    data['train'] = data['train'].map(transform_example)
    data['train_eval'] = data['train_eval'].map(transform_example)

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
    return data


def make_iterators(data, config):
    # train_iter = data['train'].map(augment_example).shuffle(1000).batch(config.batch_size, drop_remainder=True).take(-1)
    # n_train = data['train'].__len__().numpy()
    n_train = 150000
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
