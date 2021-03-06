import tensorflow as tf
import tensorflow_datasets as tfds
from scipy import signal
from data.data_parser import *
import re


def sort_data_by_label(train_data, label, validate=True):
    # sort train_data by track_id
    labels = train_data[1]
    if label == 'track_id':
        label_list = [int(label_id) for label_id in labels['{}'.format(label)]]
    elif label == 'snr_type':
        label_list = labels['{}'.format(label)]
    else:
        raise Exception('sort data supported by track/snr')
    L = [(label_list[i], i) for i in range(len(label_list))]
    L.sort()
    sorted_l, permutation = zip(*L)
    permutation = list(permutation)  # Generate list of segments id sorted by track!
    # sort the entire array
    X_sorted = train_data[0][permutation]
    labels_sorted = collections.OrderedDict()
    for key in labels.keys():
        labels_sorted[key] = [labels[key][i] for i in permutation]

    if validate:
        # validate sort:
        for i in range(X_sorted.shape[0]):
            # find index of the corresponding segment_id
            segid = labels_sorted['segment_id'][i]
            j = train_data[1]['segment_id'].index(segid)
            if np.linalg.norm(X_sorted[i] - train_data[0][j]) != 0:
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


def get_i_to_split(target_data, count):
    counter = count / 4
    imax_valid = 0
    while counter > 0:
        counter = counter - 1
        imax_valid = imax_valid + 1
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

def get_data_by_snr(snr_type, X_sorted, labels_sorted):
    target_indices = []
    for i in range(X_sorted.shape[0]):
        if labels_sorted['snr_type'][i] == snr_type:
            tid = labels_sorted['track_id'][i]
            snr_list = np.array([snr for j, snr in enumerate(labels_sorted['snr_type']) if
                        labels_sorted['track_id'][j] == tid])
            if len(np.unique(snr_list)) == 1:
                target_indices.append(i)
    X_target = X_sorted[target_indices]
    labels_target = collections.OrderedDict()
    for key in labels_sorted:
        labels_target[key] = (np.array(labels_sorted[key])[target_indices]).tolist()
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
