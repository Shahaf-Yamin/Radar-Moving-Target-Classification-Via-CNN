import tensorflow as tf
import tensorflow_datasets as tfds
from scipy import signal
from data.data_parser import *
from data.signal_processing import Sample_rectangle_from_spectrogram
import re


def sort_data_by_track(train_data,validate=True):
    # sort train_data by track_id
    labels = train_data[1]
    track_id_list = [int(tid) for tid in labels['track_id']]
    L = [(track_id_list[i], i) for i in range(len(track_id_list))]
    L.sort()
    sorted_l, permutation = zip(*L)
    permutation = list(permutation)# Generate list of segments id sorted by track!
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


def convert_numpy_to_dataset(train_data, validation_data, config):
    global g_train_data, g_validation_data,g_config

    if config.tcn_use_variable_length and bool(re.search('tcn', config.exp_name, re.IGNORECASE)):
        g_train_data = train_data
        g_validation_data = validation_data
        train_dataset = tf.data.Dataset.from_generator(ds_train_gen,(tf.float32,tf.int8))
        validation_dataset = tf.data.Dataset.from_generator(ds_validation_gen, (tf.float32, tf.int8))
    elif config.with_rect_augmentation is True:
        g_train_data = (np.expand_dims(np.array([X for X in train_data[:, 0]]), axis=-1),
                        np.expand_dims(np.array([y for y in train_data[:, 1]]), axis=-1))
        train_dataset = tf.data.Dataset.from_generator(ds_crop_generator,(tf.float32,tf.int8))
        validation_dataset = tf.data.Dataset.from_tensor_slices(
            (np.expand_dims(np.array([X for X in validation_data[:, 0]]), axis=-1),
             np.expand_dims(np.array([y for y in validation_data[:, 1]]), axis=-1)))
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
global g_train_data, g_validation_data, g_config

def ds_train_gen():
    global g_train_data
    for i in range(g_train_data.shape[0]):
        yield g_train_data[i,0], g_train_data[i,1]
def ds_validation_gen():
    global g_validation_data
    for i in range(g_validation_data.shape[0]):
        yield g_validation_data[i,0], g_validation_data[i,1]

def expand_data_generator_by_sampling_rect(X,y,config):
    new_data = []
    new_label = []
    for segment, index in zip(X, range(len(X))):
        expanded_iq = Sample_rectangle_from_spectrogram(iq_mat=segment, config=config)
        new_data.extend(expanded_iq)
        new_label_list = [y[index]] * len(expanded_iq)
        new_label.extend(new_label_list)

    return new_data,new_label

def ds_crop_generator():
    global g_config, g_train_data

    batch_features = np.zeros((g_config.batch_size, g_config.rect_augment_num_of_rows, g_config.rect_augment_num_of_cols, 1))
    batch_labels = np.zeros((g_config.batch_size, 1))

    while True:
        for batch_X,batch_y in g_train_data:
            X, y = expand_data_generator_by_sampling_rect(batch_X, batch_y, g_config)
            array_indices = np.arange(len(X))
            # for batch_repeat_index in range(config.repeat_rect_exp_per_batch):
            for i in range(g_config.batch_size):
                # choose random index in features
                index = np.random.choice(array_indices, 1)[0]
                # Delete index for the future options
                array_indices = np.delete(array_indices,index)
                # Crop the image
                batch_features[i] = X[index]
                batch_labels[i] = y[index]
            yield (batch_features, batch_labels)



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
