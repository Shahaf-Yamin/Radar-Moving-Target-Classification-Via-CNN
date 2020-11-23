import tensorflow as tf
import tensorflow_datasets as tfds
from scipy import signal
from data.data_parser import *
import re


def sort_data_by_track_id(data, validate=True):
    # sort train_data by track_id
    labels = data[1]
    track_id_list = [int(tid) for tid in labels['track_id']]
    L = [(track_id_list[i], i) for i in range(len(track_id_list))]
    L.sort()
    sorted_l, permutation = zip(*L)
    permutation = list(permutation)  # Generate list of segments id sorted by track!
    # sort the entire array
    X_sorted = data[0][permutation]
    labels_sorted = collections.OrderedDict()
    for key in labels.keys():
        labels_sorted[key] = [labels[key][i] for i in permutation]

    if validate:
        # validate sort:
        for i in range(X_sorted.shape[0]):
            # find index of the corresponding segment_id
            segid = labels_sorted['segment_id'][i]
            j = data[1]['segment_id'].index(segid)
            if np.linalg.norm(X_sorted[i] - data[0][j]) != 0:
                except_str = 'sorting missmatch! , at X_sorted[{}]'.format(i)
                raise Exception(except_str)

    return X_sorted, labels_sorted


def split_train_and_valid_by_target(target_data, imax):
    X_valid = target_data[0][:imax + 1]
    X_train = target_data[0][imax + 1:]
    labels_valid = collections.OrderedDict()
    labels_train = collections.OrderedDict()
    for key in target_data[1].keys():
        labels_valid[key] = target_data[1][key][:imax + 1]
        labels_train[key] = target_data[1][key][imax + 1:]

    return X_valid, labels_valid, X_train, labels_train


def get_i_to_split(target_data, counter):
    imax_valid = counter
    tid = target_data[1]['track_id'][imax_valid]
    imax_valid = imax_valid + 1
    while target_data[1]['track_id'][imax_valid] == tid:
        imax_valid = imax_valid + 1
    return imax_valid - 1


def get_data_by_target(target, X_sorted, labels_sorted):
    X_target = np.array([X_sorted[i] for i in range(X_sorted.shape[0]) if labels_sorted['target_type'][i] == target])
    labels_target = collections.OrderedDict()
    for key in labels_sorted:
        labels_target[key] = []
    for i in range(X_sorted.shape[0]):
        if labels_sorted['target_type'][i] == target:
            for key in labels_sorted:
                labels_target[key].append(labels_sorted[key][i])

    return X_target, labels_target


def get_data_by_snr(snr_type, X, labels):
    target_indices = np.bool_(np.zeros(len(X)))
    # arrays for snr_type and track_id
    snr_arr = np.array(labels['snr_type'])
    track_arr = np.array(labels['track_id'])
    # get index list of all appearnaces of this snr
    snr_indices = np.where(snr_arr == snr_type)[0].tolist()
    target_indices[snr_indices] = True
    # get track id array of all tracks with this snr
    tid_with_target_snr = track_arr[snr_indices]
    for tid in np.unique(tid_with_target_snr):
        # find indices of this track id
        tid_index_list = np.where(track_arr == tid)
        if len(np.unique(snr_arr[tid_index_list])) > 1:
            # False the target_indices with two types of snr
            target_indices[tid_index_list] = False

    # return examples + labelsfrom target_indices
    X_target = X[target_indices]
    labels_target = collections.OrderedDict()
    for key in labels.keys():
        labels_target[key] = (np.array(labels[key])[target_indices]).tolist()
    # append
    return (X_target, labels_target), target_indices


def convert_numpy_to_dataset(train_data, validation_data, config):
    global g_train_data, g_validation_data

    if config.tcn_use_variable_length and bool(re.search('tcn', config.exp_name, re.IGNORECASE)):
        g_train_data = train_data
        g_validation_data = validation_data
        train_dataset = tf.data.Dataset.from_generator(ds_train_gen, (tf.float32, tf.int8))
        validation_dataset = tf.data.Dataset.from_generator(ds_validation_gen, (tf.float32, tf.int8))
    elif config.learn_background:
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (np.expand_dims(np.array([X for X in train_data[:, 0]]), axis=-1),
             np.array([y for y in train_data[:, 1]])))
        validation_dataset = tf.data.Dataset.from_tensor_slices(
            (np.expand_dims(np.array([X for X in validation_data[:, 0]]), axis=-1),
             np.array([y for y in validation_data[:, 1]])))
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (np.expand_dims(np.array([X for X in train_data[:, 0]]), axis=-1),
             np.expand_dims(np.array([y for y in train_data[:, 1]]), axis=-1)))
        validation_dataset = tf.data.Dataset.from_tensor_slices(
            (np.expand_dims(np.array([X for X in validation_data[:, 0]]), axis=-1),
             np.expand_dims(np.array([y for y in validation_data[:, 1]]), axis=-1)))

    data = {'train': train_dataset, 'train_eval': validation_dataset}
    return data


def do_nothing(iq_mat, label):
    return iq_mat, label


# WA for working with variable length dataset in the TCN
global g_train_data, g_validation_data


def ds_train_gen():
    global g_train_data
    for i in range(g_train_data.shape[0]):
        yield g_train_data[i, 0], g_train_data[i, 1]


def ds_validation_gen():
    global g_validation_data
    for i in range(g_validation_data.shape[0]):
        yield g_validation_data[i, 0], g_validation_data[i, 1]


def convert_metadata_to_numpy(train_data, validation_data):
    return np.array([do_nothing(iq_mat, label) for iq_mat, label in zip(train_data[0], train_data[1]['target_type'])]), \
           np.array([do_nothing(iq_mat, label) for iq_mat, label in
                     zip(validation_data[0], validation_data[1]['target_type'])])


def validate_train_val_tracks(train_tracks, valid_tracks):
    count = 0
    for train_tid in train_tracks:
        if train_tid in valid_tracks:
            count = count + 1

    if count > 0:
        raise Exception('Validation set and Train set contain {} corresponding tracks!'.format(count))


# def convert_data_to_sequential(data_iterators):
#     def reshape_exapmle(iq_mat, label):
#         tf.transpose(iq_mat)
#         return iq_mat, label
#
#     # swap axes
#     train_ds = data_iterators['train']
#     valid_ds = data_iterators['train_eval']
#     train_ds = train_ds.map(lambda iq_mat, label: (reshape_exapmle(iq_mat, tf.float32), label))
#
#     X_train = train_data[0].swapaxes(1, 2)
#     X_val = validation_data[0].swapaxes(1, 2)
#     # build tuples
#     train_data = (X_train, train_data[1])
#     validation_data = (X_val, validation_data[1])
#     return train_data, validation_data

def lstm_preprocess_data(train_data, validation_data, config):
    train_data = np.array([(np.swapaxes(sample[0], axis1=0, axis2=1), sample[1]) for sample in train_data])
    validation_data = np.array(
        [(np.swapaxes(sample[0], axis1=0, axis2=1), sample[1]) for sample in validation_data])

    return train_data, validation_data


def tcn_preprocess_data(train_data, validation_data, config):
    def flatten_example(iq_mat, label):
        return iq_mat.flatten(), label

    def up_sample_example(iq_mat, label, up):
        iq_mat = signal.resample_poly(iq_mat, up, down=1, axis=0)
        return iq_mat, label

    if config.tcn_flattend is True:
        train_data = np.array([flatten_example(sample[0], sample[1]) for sample in train_data])
        validation_data = np.array([flatten_example(sample[0], sample[1]) for sample in validation_data])
    else:
        train_data = np.array([(np.swapaxes(sample[0], axis1=0, axis2=1), sample[1]) for sample in train_data])
        validation_data = np.array(
            [(np.swapaxes(sample[0], axis1=0, axis2=1), sample[1]) for sample in validation_data])

    if config.tcn_upsample_slow_axis is True:
        up = config.tcn_upsample_factor
        train_data = np.array([up_sample_example(sample[0], sample[1], up) for sample in train_data])
        validation_data = np.array([up_sample_example(sample[0], sample[1], up) for sample in validation_data])
        model_input_dim = config.model_input_dim
        model_input_dim[0] = model_input_dim[0] * up
        config.__setattr__("model_input_dim", model_input_dim)

    return train_data, validation_data


def reshape_label(data):
    t = data[:, 1]
    target_arr = np.zeros((t.size, 2))
    target_arr[:, 0] = t
    target_arr[:, 1] = np.array([1 if y == 2 else 0 for y in t])
    data = np.array([(iq_mat, label) for iq_mat, label in
                     zip(data[:, 0], target_arr)])

    # validate
    _1 = [y[0] == 2 and y[1] != 1 for y in data[:, 1]]
    _2 = [y[0] != 2 and y[1] != 0 for y in data[:, 1]]
    assert not (True in _1 or True in _2)

    return data


def noise_human_validation_data(validation_data):
    human_indices = np.where(np.array(validation_data[1]['target_type']) == 1)[0]
    indices = np.random.choice(human_indices, len(human_indices) // 2, replace=False)
    for i in indices:
        validation_data[0][i] = validation_data[0][i] + np.random.normal(loc=0, scale=0.75,
                                                                         size=validation_data[0][i].shape)
        validation_data[1]['snr_type'][i] = 'LowSNR'
    val_snr = np.array(validation_data[1]['snr_type'])
    val_target = np.array(validation_data[1]['target_type'])
    animal_low_snr = len(np.intersect1d(np.where(val_snr=='LowSNR') , np.where(val_target==0)))
    animal_high_snr = len(np.intersect1d(np.where(val_snr=='HighSNR') , np.where(val_target==0)))
    human_low_snr = len(np.intersect1d(np.where(val_snr=='LowSNR') , np.where(val_target==1)))
    human_high_snr = len(np.intersect1d(np.where(val_snr=='HighSNR') , np.where(val_target==1)))
    print(20 * '#')
    print('validation set distribution:')
    print('animal_low_snr = {}'.format(animal_low_snr))
    print('animal_high_snr = {}'.format(animal_high_snr))
    print('human_low_snr = {}'.format(human_low_snr))
    print('human_high_snr = {}'.format(human_high_snr))
    print('val_snr low count : {}, val_snr high count: {}'.format(len(val_snr[val_snr == 'LowSNR']),
                                                                  len(val_snr[val_snr == 'HighSNR'])))
    print('val_target animal count : {}, val_target human count: {}'.format(len(val_target[val_target == 0]),
                                                                            len(val_target[val_target == 1])))
    return validation_data


def split_train_and_valid_by_snr(snr_type, X, labels, count_valid_animal, count_valid_human,all_snr):
    assert (snr_type == 'HighSNR') or (snr_type == 'LowSNR')
    opposite_snr_type = 'HighSNR' if snr_type == 'LowSNR' else 'LowSNR'
    # get_data_by_snr() returns indices and data with NO SNR duality, i.e. can be only Low/High not both
    data, indices = get_data_by_snr(snr_type=snr_type, X=X, labels=labels)
    assert (not (opposite_snr_type in data[1]['snr_type']))

    # sort data by track_id
    X_sorted, labels_sorted = sort_data_by_track_id(data)

    # split to animal data
    animal_data = get_data_by_target(target=0, X_sorted=X_sorted, labels_sorted=labels_sorted)
    assert  not (1 in animal_data[1]['target_type'])
    # imax is the last index (INCLUDING!) of the validation data
    imax_valid_animal = get_i_to_split(target_data=animal_data, counter=count_valid_animal)

    # animal data
    X_valid_animal, labels_valid_animal, X_train_animal, labels_train_animal = split_train_and_valid_by_target(
        target_data=animal_data, imax=imax_valid_animal)

    # initiate train and valid data with animal, since it will be in both SNRs
    labels_train = collections.OrderedDict()
    labels_valid = collections.OrderedDict()
    X_train = X_train_animal
    X_valid = X_valid_animal
    for key in labels.keys():
        labels_train[key] = labels_train_animal[key]
        labels_valid[key] = labels_valid_animal[key]

    if snr_type != 'LowSNR':
        # human data - Only 88 segments of human LowSNR are in train.
        # split to animal data
        human_data = get_data_by_target(target=1, X_sorted=X_sorted, labels_sorted=labels_sorted)
        assert not (0 in human_data[1]['target_type'])
        # imax is the last index (INCLUDING!) of the validation data
        imax_valid_human = get_i_to_split(target_data=human_data, counter=count_valid_human)

        X_valid_human, labels_valid_human, X_train_human, labels_train_human = split_train_and_valid_by_target(
            target_data=human_data, imax=imax_valid_human)
        X_train = np.concatenate((X_train, X_train_human), axis=0)
        X_valid = np.concatenate((X_valid, X_valid_human), axis=0)
        for key in labels.keys():
            labels_train[key].extend(labels_train_human[key])
            labels_valid[key].extend(labels_valid_human[key])
    else:
        if all_snr is False:
            human_data = get_data_by_target(target=1, X_sorted=X_sorted, labels_sorted=labels_sorted)
            assert not (0 in human_data[1]['target_type'])
            X_train = np.concatenate((X_train, human_data[0]), axis=0)
            for key in labels.keys():
                labels_train[key].extend(human_data[1][key])

    return X_valid, labels_valid, X_train, labels_train, indices


def print_data_distribution_before_transform_to_dataset(train_data,validation_data):
    """ label distribution to print in logs """
    # print_target_distribution(train_data, validation_data)
    print(20 * '#')
    print('Data distribution BEFORE conversion to tf.data.Dataset')
    print('No. segments in train: {}'.format(len(train_data)))
    print('No. segments in validation: {}'.format(len(validation_data)))
    print('train targets: {} humans, {} animals'.format(len(train_data[:, 1][train_data[:, 1] == 1]),
                                                        len(train_data[:, 1][train_data[:, 1] == 0])))
    print('validation targets: {} humans, {} animals'.format(len(validation_data[:, 1][validation_data[:, 1] == 1]),
                                                             len(validation_data[:, 1][validation_data[:, 1] == 0])))
