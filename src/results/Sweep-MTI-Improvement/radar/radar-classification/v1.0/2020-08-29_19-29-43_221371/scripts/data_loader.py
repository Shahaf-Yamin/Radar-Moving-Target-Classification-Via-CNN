import copy
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
        if num_of_tid > 11:
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
def expand_data_by_interpolation(data_parser,train_data):

    interpolated_data = data_parser.get_interpolation_data()
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

def expand_human_data_by_aux_set(data_parser,train_data):
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
    if abs(num_of_animals / (num_of_animals + num_of_humans) - 0.5) > 0.05:
        raise Exception('Using stable mode and data resulted as unstable !!')
    return X, labels



def load_data(config):
    train_data, validation_data, data_parser = read_data(config)

    if config.augment_by_track is True:  # augment_by_track the human examples
        train_data = expand_human_data_by_tracks(train_data=train_data, config=config)

    if config.drop_geolocation is True:
        train_data = drop_geolocation(train_data=train_data,config=config)

    if config.stable_mode is True:
        train_data = expand_human_data_by_aux_set(data_parser=data_parser, train_data=train_data)

    if config.with_interpolation is True:
        train_data = expand_data_by_interpolation(data_parser=data_parser, train_data=train_data)

    train_data, validation_data = convert_metadata_to_numpy(train_data, validation_data)

    train_data = augment(train_data, config)

    data = convert_numpy_to_dataset(train_data, validation_data)

    transformed_data = transform(data, config)

    data_iterators = make_iterators(transformed_data, config)

    # swap axes for sequential Data
    if config.exp_name == "LSTM":
        data_iterators = convert_data_to_sequential(data_iterators)

    return data_iterators


def read_data(config):
    stable_mode = config.get('stable_mode')
    testset_size = config.get('N_test')
    data_parser = DataSetParser(stable_mode=stable_mode,config=config)

    data_parser.plot_single_timestamp_frequency()
    data_parser.plot_IQ_data()

    train_data = data_parser.get_dataset_allsnr(dataset_type='train')
    # sort train_data by track id to avoid over fitting over the validation data
    train_data, validation_data = generate_test_set(train_data, testset_size)

    # validate tracks
    validate_train_val_tracks(train_tracks=train_data[1]['track_id'], valid_tracks=validation_data[1]['track_id'])

    return train_data, validation_data, data_parser


def transform(data, config):
    def transform_example(iq_mat, label):
        iq_mat, label = tf.cast(iq_mat, tf.float32), tf.cast(label, tf.int8)
        return iq_mat, label

    data['train'] = data['train'].map(transform_example)
    data['train_eval'] = data['train_eval'].map(transform_example)

    return data

def augment(data, config):

    def augment_normal(iq_mat, label):
        iq_mat = iq_mat + np.random.normal(loc=config.augment_normal_mean,scale=config.augment_normal_std,size=iq_mat.shape)
        return iq_mat, label

    def augment_gaussian_filt_2d(iq_mat, label):
        iq_mat = gaussian_filter(iq_mat, sigma=1)
        return iq_mat, label

    def augment_freq_shift(iq_mat,label):
        if np.random.binomial(1,0.5,size=1)[0] == 0:
            shift = config.freq_shift_delta
        else:
            shift = -config.freq_shift_delta
        if config.shift_freq_dc_width == 0:
            return np.roll(iq_mat, shift,axis=0), label
        else:
            shiftwidth = config.shift_freq_dc_width
            iq_mat[shiftwidth:-shiftwidth,:] = np.roll(iq_mat[shiftwidth:-shiftwidth,:],shift, axis=0)
            return iq_mat, label

    def augment_timestep_shift(iq_mat,label):
        if np.random.binomial(1,0.5,size=1)[0] == 0:
            shift = config.timestep_shift_delta
        else:
            shift = -config.timestep_shift_delta
        return np.roll(iq_mat, shift,axis=1), label

    def augment_vertical_flip(iq_mat,label):
        return np.flipud(iq_mat), label

    def augment_horiz_flip(iq_mat,label):
        return np.fliplr(iq_mat), label

    orig_data = data
    augment_funcs = config.augment_funcs
    for augment_index in range(config.augment_expansion_number):
        if 'flip_image' in augment_funcs:
            data = np.concatenate((data, np.array([augment_vertical_flip(sample[0], sample[1]) for sample in orig_data])), axis=0)
        if 'horiz_flip' in augment_funcs:
            data = np.concatenate((data, np.array([augment_horiz_flip(sample[0], sample[1]) for sample in orig_data])), axis=0)
        if 'normal' in augment_funcs:
            data = np.concatenate((data, np.array([augment_normal(sample[0],sample[1]) for sample in orig_data])),axis=0)
        if 'guassian_filt' in augment_funcs:
            data = np.concatenate((data, np.array([augment_gaussian_filt_2d(sample[0],sample[1]) for sample in orig_data])),axis=0)
        if 'freq_shift' in augment_funcs:
            data = np.concatenate((data, np.array([augment_freq_shift(sample[0],sample[1]) for sample in orig_data])),axis=0)
        if 'timestep_shift' in augment_funcs:
            data = np.concatenate((data, np.array([augment_timestep_shift(sample[0],sample[1]) for sample in orig_data])),axis=0)
    return data


def make_iterators(data, config):
    # train_iter = data['train'].map(augment_example).shuffle(1000).batch(config.batch_size, drop_remainder=True).take(-1)
    train_iter = data['train'].shuffle(1000).batch(config.batch_size, drop_remainder=True).take(-1)
    train_eval_iter = data['train_eval'].batch(config.batch_size_eval).take(-1)

    iterators = {'train': train_iter,
                 'train_eval': train_eval_iter}
    return iterators

def drop_geolocation(train_data,config):
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

    return X , labels



