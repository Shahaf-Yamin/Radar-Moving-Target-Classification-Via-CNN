import tensorflow as tf
import tensorflow_datasets as tfds
from data.data_parser import *
from data.data_loader_utils import *
from scipy.ndimage import gaussian_filter


def generate_test_set(train_data, TRACK_ID_TEST_SET_SIZE):
    """
    Since test date given is un-labeled, in order to evaluate the the model properly we need to hold-out training
    examples from the train set to be used as test.
    The train set is divided into tracks so we need to make sure there are no segments from same tracks in train & test
    to avoid inherited over-fitting.
    """
    # sort train_data by track_id
    X_sorted, labels_sorted = sort_data_by_track(train_data)

    # save for validation after split
    train_data_orig = train_data

    # split to human/animal data
    human_data = get_data_by_target(target=1, X_sorted=X_sorted, labels_sorted=labels_sorted)
    animal_data = get_data_by_target(target=0, X_sorted=X_sorted, labels_sorted=labels_sorted)
    # imax is the last index (INCLUDING!) of the validation data
    imax_valid_animal = get_i_to_split(target_data=animal_data, count=TRACK_ID_TEST_SET_SIZE)
    imax_valid_human = get_i_to_split(target_data=human_data, count=TRACK_ID_TEST_SET_SIZE)

    # human data
    X_valid_human, labels_valid_human, X_train_human, labels_train_human = split_train_and_valid_by_target(
        target_data=human_data, imax=imax_valid_human)
    # animal data
    X_valid_animal, labels_valid_animal, X_train_animal, labels_train_animal = split_train_and_valid_by_target(
        target_data=animal_data, imax=imax_valid_animal)

    # split train and validation total
    X_valid = np.concatenate((X_valid_human, X_valid_animal), axis=0)
    X_train = np.concatenate((X_train_human, X_train_animal), axis=0)
    labels_train = collections.OrderedDict()
    labels_valid = collections.OrderedDict()
    for key in labels_sorted.keys():
        labels_valid[key] = labels_valid_human[key]
        labels_valid[key].extend(labels_valid_animal[key])
        labels_train[key] = labels_train_human[key]
        labels_train[key].extend(labels_train_animal[key])

    for i in range(X_train.shape[0]):
        if np.linalg.norm(X_train[i] - train_data_orig[0][int(labels_train['segment_id'][i])]) != 0:
            raise Exception('sorting missmatch! , at X_train[{}]'.format(i))
    for i in range(X_valid.shape[0]):
        if np.linalg.norm(X_valid[i] - train_data_orig[0][int(labels_valid['segment_id'][i])]) != 0:
            raise Exception('sorting missmatch! , at X_valid[{}]'.format(i))

    train_data = (X_train, labels_train)
    validation_data = (X_valid, labels_valid)
    return train_data, validation_data


def expand_human_data_by_tracks(train_data, config):
    # sort train_data by track_id
    X_sorted, labels_sorted = sort_data_by_track(train_data)
    X_augmented = []
    augment_count = 0
    labels_augmented = collections.OrderedDict()
    for key in labels_sorted.keys():
        labels_augmented[key] = []
    offset = config.augment_by_track_offset
    tid_int_list = [int(tid) for tid in labels_sorted['track_id']]
    tid_unique_list = np.unique(np.array(tid_int_list)).tolist()
    for tid in tid_unique_list:
        i_list = [i for i, x in enumerate(tid_int_list) if x == tid]
        num_of_tid = len(i_list)
        if num_of_tid > 8:
            step_size = 2 * 32
        else:
            step_size = 32
        X_tid = np.concatenate(X_sorted[i_list], axis=1)
        i = offset
        augment_local_count = 0
        while i < X_tid.shape[1] - 32 and augment_local_count < config.augment_by_track_local_count:
            X_augmented.append(X_tid[:, i:i + 32])
            for key in labels_augmented.keys():
                if key == 'segment_id':
                    labels_augmented[key].append(str(-(augment_count + 1)))
                else:
                    labels_augmented[key].append(labels_sorted[key][i_list[0]])
            i = i + step_size
            augment_local_count = augment_local_count + 1
            augment_count = augment_count + 1

    X_augmented = np.stack(X_augmented, axis=0)
    X_train_new = np.concatenate((X_sorted, X_augmented), axis=0)
    labels_new = collections.OrderedDict()
    for key in labels_sorted.keys():
        labels_new[key] = labels_sorted[key]
        labels_new[key].extend(labels_augmented[key])

    return X_train_new, labels_new


def load_data(config):
    train_data, validation_data = read_data(config)

    if config.augment_by_track is True:  # augment_by_track the human examples
        train_data = expand_human_data_by_tracks(train_data=train_data, config=config)

    data = convert_numpy_to_dataset(train_data, validation_data)

    transformed_data = transform(data, config)

    augmented_data = augment(transformed_data, config)

    data_iterators = make_iterators(augmented_data, config)

    # swap axes for sequential Data
    if config.exp_name == "LSTM":
        data_iterators = convert_data_to_sequential(data_iterators)

    return data_iterators


def read_data(config):
    stable_mode = config.get('stable_mode')
    testset_size = config.get('N_test')
    data_parser = DataSetParser(stable_mode=stable_mode)
    train_data = data_parser.get_dataset_allsnr(dataset_type='train')
    # sort train_data by track id to avoid over fitting over the validation data
    train_data, validation_data = generate_test_set(train_data, testset_size)
    if stable_mode is True:
        train_data_aux_exp = data_parser.aux_split(config.segments_per_aux_track)
        X = np.concatenate((train_data[0], train_data_aux_exp[0]), axis=0)  # Stack the experiment data
        train_labels = train_data[1]
        train_aux_exp_labels = train_data_aux_exp[1]
        for key in train_labels.keys():
            train_labels[key].extend(train_aux_exp_labels[key])
        labels = train_labels
        train_data = (X, labels)

    # validate tracks
    validate_train_val_tracks(train_tracks=train_data[1]['track_id'], valid_tracks=validation_data[1]['track_id'])

    return train_data, validation_data


def transform(data, config):
    def transform_example(iq_mat, label):
        iq_mat, label = tf.cast(iq_mat, tf.float32), tf.cast(label, tf.int8)
        return iq_mat, label

    data['train'] = data['train'].map(transform_example)
    data['train_eval'] = data['train_eval'].map(transform_example)

    return data


def augment(data, config):
    def augment_normal(iq_mat, label):
        iq_mat = iq_mat + tf.random.normal(shape=tf.shape(iq_mat), mean=config.augment_normal_mean,
                                           stddev=config.augment_normal_std)
        return iq_mat, label

    def augment_gaussian_filt_2d(iq_mat, label):
        iq_mat = tf.convert_to_tensor(gaussian_filter(iq_mat.numpy(), sigma=1))
        return iq_mat, label
    org_train_data = data['train']
    augment_func_list = config.augment_func_list
    if 'normal' in augment_func_list:
        data['train'] = data['train'].concatenate(org_train_data.map(lambda iq_mat, label: (augment_normal(iq_mat, label))))
    if 'guassian_filt' in augment_func_list:
        data['train'] = data['train'].concatenate(org_train_data.map(lambda iq_mat, label: (augment_gaussian_filt_2d(iq_mat, label))))
    if 'flip_image' in augment_func_list:
        data['train'] = data['train'].concatenate(org_train_data.map(lambda iq_mat, label: (tf.image.flip_up_down(iq_mat), label)))
    data['train'] = data['train'].repeat(config.augment_repeat)
    data['train'].cache()
    return data


def make_iterators(data, config):
    # train_iter = data['train'].map(augment_example).shuffle(1000).batch(config.batch_size, drop_remainder=True).take(-1)
    train_iter = data['train'].shuffle(1000).batch(config.batch_size, drop_remainder=True).take(-1)
    train_eval_iter = data['train_eval'].batch(config.batch_size_eval).take(-1)

    iterators = {'train': train_iter,
                 'train_eval': train_eval_iter}
    return iterators
