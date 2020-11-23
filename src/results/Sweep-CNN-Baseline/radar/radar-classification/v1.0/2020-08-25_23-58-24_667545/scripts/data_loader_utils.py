import tensorflow as tf
import tensorflow_datasets as tfds
from data.data_parser import *


def sort_data_by_track(train_data):
    # sort train_data by track_id
    labels = train_data[1]
    track_id_list = [int(tid) for tid in labels['track_id']]
    L = [(track_id_list[i], i) for i in range(len(track_id_list))]
    L.sort()
    sorted_l, permutation = zip(*L)
    permutation = list(permutation)
    # sort the entire array
    X_sorted = train_data[0][permutation]
    labels_sorted = collections.OrderedDict()
    for key in labels.keys():
        labels_sorted[key] = [labels[key][i] for i in permutation]

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
    counter = count / 2
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


def convert_numpy_to_dataset(train_data, validation_data):
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


def convert_metadata_to_numpy(train_data, validation_data):
    return np.array([do_nothing(iq_mat, label) for iq_mat, label in zip(train_data[0], train_data[1]['target_type'])]), \
           np.array([do_nothing(iq_mat, label) for iq_mat, label in zip(validation_data[0], validation_data[1]['target_type'])])


def validate_train_val_tracks(train_tracks, valid_tracks):
    count = 0
    for train_tid in train_tracks:
        if train_tid in valid_tracks:
            count = count + 1

    if count > 0:
        raise Exception('Validation set and Train set contain {} corresponding tracks!'.format(count))


def convert_data_to_sequential(data_iterators):
    def reshape_exapmle(iq_mat, label):
        tf.transpose(iq_mat)
        return iq_mat, label

    # swap axes
    train_ds = data_iterators['train']
    valid_ds = data_iterators['train_eval']
    train_ds = train_ds.map(lambda iq_mat, label: (reshape_exapmle(iq_mat, tf.float32), label))

    X_train = train_data[0].swapaxes(1, 2)
    X_val = validation_data[0].swapaxes(1, 2)
    # build tuples
    train_data = (X_train, train_data[1])
    validation_data = (X_val, validation_data[1])
    return train_data, validation_data
