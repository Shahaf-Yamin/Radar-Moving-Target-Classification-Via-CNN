from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, Softmax, BatchNormalization, Dropout, LSTM, \
    AveragePooling2D, Input, Bidirectional, SpatialDropout1D, LayerNormalization, Activation, LeakyReLU, Conv1D, \
    MaxPooling1D, Conv3D, MaxPooling3D
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from models import res_net, tcn
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Activation, Multiply, Add, Lambda, Layer
import tensorflow.keras.initializers
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.losses import BinaryCrossentropy
import os
import ssl

background_implicit_inference = False


class BlockBackgroundModel(Model):
    """ https://keras.io/guides/customizing_what_happens_in_fit/ """

    def test_step(self, data):
        global background_implicit_inference
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred_model = self(x, training=False)
        if background_implicit_inference:
            a = tf.map_fn(lambda x: x[0], elems=y_pred_model)
            b = tf.map_fn(lambda x: x[1] + x[2], elems=y_pred_model)
        else:
            a = tf.map_fn(lambda x: x[0] / (1 - x[2]), elems=y_pred_model)
            b = tf.map_fn(lambda x: x[1] / (1 - x[2]), elems=y_pred_model)
        y_pred = tf.transpose(tf.stack([a, b]))
        # validate prediction <= 1
        y_pred = tf.map_fn(lambda x: tf.cond(tf.greater(x[0] + x[1], 1.0), lambda: x / (x[0] + x[1]), lambda: x),
                           elems=y_pred)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


class complex_activation(Layer):
    def call(self, inputs, **kwargs):
        return tf.math.sqrt(tf.math.add(tf.math.pow(inputs[:, :, :, :, 0], 2), tf.math.pow(inputs[:, :, :, :, 1], 2)))

def adjust_input_size(config):
    config.__setattr__("model_input_dim", [126 * config.freq_expansion_interpolation, 32, 2])
    if config.with_iq_matrices is True or config.with_magnitude_phase is True:
        config.__setattr__("model_input_dim", [126 * config.freq_expansion_interpolation, 32, 2])
    if config.with_rect_augmentation:
        config.__setattr__("model_input_dim", [126 * config.freq_expansion_interpolation,config.rect_augment_num_of_timesteps, 1])
    if config.with_rect_augmentation and (config.with_iq_matrices or config.with_magnitude_phase):
        config.__setattr__("model_input_dim", [126 * config.freq_expansion_interpolation,config.rect_augment_num_of_timesteps, 2])
    if config.use_color_map_representation is True:
        config.__setattr__("model_input_dim", [126 * config.freq_expansion_interpolation, 32, 3])

    assert not (config.freq_expansion_interpolation != 32 and (config.with_iq_matrices or config.with_magnitude_phase or config.with_rect_augmentation))

    if config.with_slow_time_interpolation is True:
        model_input_dim = config.model_input_dim
        config.__setattr__("model_input_dim", [model_input_dim[0], config.slow_time_interpolation, 2])

    return config


def build_model(config):
    if config.model_name == "radar-cnn":
        model = {'train': CNN_BNModel(config)}
    elif config.model_name == "cnn-baseline":
        model = {'train': BaselineCNNModel(config)}
    elif config.model_name == "VGG19":
        model = {'train': VGG19Model(config)}
    elif config.model_name == "VGG16":
        model = {'train': VGG16Model(config)}
    elif config.model_name == "resnet50":
        model = {'train': Resnet50Model(config)}
    elif config.model_name == "radar-lstm":
        model = {'train': LSTMModel(config)}
    elif config.model_name == "radar-resnet":
        model = {'train': RESNETModel(config)}
    elif config.model_name == "radar-CNN-Extended":
        model = {'train': CNNModel(config)}
    elif config.model_name == "radar-CNN-ResBlock":
        model = {'train': CNNResBlockModel(config)}
    elif config.model_name == "tcn":
        model = {'train': tcn.get_tcn_model(config)}
    elif config.model_name == 'Complex-ResBlock':
        model = {'train': CNNResBlockModelComplex(config)}
    elif config.model_name == 'Complex-Baseline':
        model = {'train': ComplexBaselineCNNModel(config)}
    else:
        raise ValueError("'{}' is an invalid model name")

    return model

def build_or_load_model(config):
    if config.load_model_weights_from_file:
        print('CURRENT DIR: {}'.format(os.getcwd()))
        adjust_input_size(config)
        model_dict = build_model(config)
        model_dict['train'].load_weights(config.model_weights_file)
        model = {'train': model_dict['train']}
        model['train'].compile(optimizer=Adam(learning_rate=config.learning_rate), loss=BinaryCrossentropy(),
                      metrics=['accuracy', AUC()])
    elif config.load_complete_model_from_file:
        model = {'train': tf.keras.models.load_model(config.complete_model_file)}
    else:
        model = build_model(config)

    model['train'].summary()
    return model

def CNN_BNModel(config):
    lamda = config.Regularization_term
    filter_shape = (config.Filter_shape_dim1[0], config.Filter_shape_dim2[0])
    filters_number = config.hidden_size[0]
    pool_shape = (config.Pool_shape_dim1[0], config.Pool_shape_dim2[0])
    input_shape = config.model_input_dim
    dense_size = config.Dense_size[0]
    p_dropout = config.dropout
    inputs = keras.Input(shape=config.model_input_dim)
    x = Conv2D(filters_number, filter_shape, activation='relu', kernel_regularizer=keras.regularizers.l2(lamda),
               bias_regularizer=keras.regularizers.l2(lamda), input_shape=input_shape)(inputs)
    x = BatchNormalization()(x)
    if p_dropout != 0:
        x = Dropout(p_dropout)(x)
    x = MaxPooling2D(pool_shape[0], pool_shape[1])(x)
    x = Flatten()(x)
    x = Dense(dense_size, activation='relu', kernel_regularizer=keras.regularizers.l2(lamda),
              bias_regularizer=keras.regularizers.l2(lamda))(x)
    if p_dropout != 0:
        x = Dropout(p_dropout)(x)
    x = BatchNormalization()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def ComplexBaselineCNNModel(config):
    lamda = config.Regularization_term
    filt1 = (config.Filter_shape_dim1[0], config.Filter_shape_dim2[0])
    filt2 = (config.Filter_shape_dim1[1], config.Filter_shape_dim2[1])
    size1 = config.hidden_size[0]
    size2 = config.hidden_size[1]
    pool_shape1 = (config.Pool_shape_dim1[0], config.Pool_shape_dim2[0])
    pool_shape2 = (config.Pool_shape_dim1[1], config.Pool_shape_dim2[1])
    input_shape = config.model_input_dim
    dense_size1 = config.Dense_size[0]
    dense_size2 = config.Dense_size[1]
    p_dropout = config.dropout

    input_layer = Input(shape=input_shape)

    x = input_layer
    real_part = tf.expand_dims(x[:, :, :, 0], axis=-1)
    imag_part = tf.expand_dims(x[:, :, :, 1], axis=-1)

    real_part_output = Conv2D(32, kernel_size=(3, 3), padding='same')(real_part)
    imag_part_output = Conv2D(32, kernel_size=(3, 3), padding='same')(imag_part)
    real = tf.expand_dims(real_part_output, axis=-1)
    imag = tf.expand_dims(imag_part_output, axis=-1)
    filter_output = tf.concat([real, imag], axis=-1)
    mixed_output = complex_activation()(filter_output)
    mixed_output = tf.expand_dims(mixed_output, axis=-1)

    real = LeakyReLU(alpha=0.3)(real)
    imag = LeakyReLU(alpha=0.3)(imag)
    layer1_output = tf.concat([real, imag, mixed_output], axis=3)[:, :, :, :, 0]

    # real_part = tf.expand_dims(layer1_output[:, :, :, 0], axis=-1)
    # imag_part = tf.expand_dims(layer1_output[:, :, :, 1], axis=-1)
    # mixed_part = tf.expand_dims(layer1_output[:, :, :, 2], axis=-1)

    x = layer1_output

    # real_part_output = Conv2D(32, kernel_size=(3, 3), padding='same')(real_part)
    # imag_part_output = Conv2D(32, kernel_size=(3, 3), padding='same')(imag_part)
    # real = tf.expand_dims(real_part_output, axis=-1)
    # imag = tf.expand_dims(imag_part_output, axis=-1)
    # filter_output = tf.concat([real, imag], axis=-1)
    # mixed_output = complex_activation()(filter_output)
    # mixed_output = tf.expand_dims(mixed_output, axis=-1)

    x = Conv2D(size1, kernel_size=filt1, activation='relu', bias_regularizer=keras.regularizers.l2(lamda),
               kernel_regularizer=keras.regularizers.l2(lamda), input_shape=input_shape)(x)
    x = MaxPooling2D(pool_size=(5, 5))(x)
    x = Conv2D(size2, kernel_size=filt2, activation='relu', bias_regularizer=keras.regularizers.l2(lamda),
               kernel_regularizer=keras.regularizers.l2(lamda))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(dense_size1, kernel_regularizer=keras.regularizers.l2(lamda), activation='relu')(x)
    x = Dense(dense_size2, kernel_regularizer=keras.regularizers.l2(lamda), activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)

    output_layer = x
    model = Model(input_layer, output_layer)

    return model


def BaselineCNNModel(config):
    lamda = config.Regularization_term
    filt1 = (config.Filter_shape_dim1[0], config.Filter_shape_dim2[0])
    filt2 = (config.Filter_shape_dim1[1], config.Filter_shape_dim2[1])
    size1 = config.hidden_size[0]
    size2 = config.hidden_size[1]
    pool_shape1 = (config.Pool_shape_dim1[0], config.Pool_shape_dim2[0])
    pool_shape2 = (config.Pool_shape_dim1[1], config.Pool_shape_dim2[1])
    input_shape = config.model_input_dim
    dense_size1 = config.Dense_size[0]
    dense_size2 = config.Dense_size[1]
    p_dropout = config.dropout

    model = Sequential()
    model.add(Conv2D(size1, kernel_size=filt1, activation='relu', bias_regularizer=keras.regularizers.l2(lamda),
                     kernel_regularizer=keras.regularizers.l2(lamda), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=pool_shape1))
    model.add(Conv2D(size2, kernel_size=filt2, activation='relu', bias_regularizer=keras.regularizers.l2(lamda),
                     kernel_regularizer=keras.regularizers.l2(lamda)))
    model.add(MaxPooling2D(pool_size=pool_shape2))
    model.add(Flatten())
    if config.dropout_after_all_conv2d > 0.0:
        model.add(Dropout(rate=config.dropout_after_all_conv2d))
    model.add(Dense(dense_size1, kernel_regularizer=keras.regularizers.l2(lamda), activation='relu'))
    model.add(Dense(dense_size2, kernel_regularizer=keras.regularizers.l2(lamda), activation='relu'))
    if p_dropout > 0.0:
        model.add(Dropout(rate=p_dropout))
    model.add(Dense(1, activation='sigmoid'))
    return model


def CNNResBlockModel(config):
    def regularization(lamda):
        if config.regularization_method == 'L2':
            return keras.regularizers.l2(lamda)
        elif config.regularization_method == 'L1':
            return keras.regularizers.l1(lamda)
        else:
            raise Exception('Use Only L2 / L1 regularization')
    def activation(activation_name, x):
        if activation_name == 'leaky_relu':
            return LeakyReLU(alpha=config.alpha)(x)
        else:
            return Activation(activation_name)(x)

    def highway_layer(value, gate_bias=-3):
        # https://towardsdatascience.com/review-highway-networks-gating-function-to-highway-image-classification-5a33833797b5
        nonlocal i_hidden  # to keep i_hidden "global" to all functions under CNNResBlockModel()
        dim = K.int_shape(value)[-1]
        # gate_bias_initializer = tensorflow.keras.initializers.Constant(gate_bias)
        # gate = Dense(units=dim, bias_initializer=gate_bias_initializer)(value)
        # gate = Activation("sigmoid")(gate)
        # TODO (just for yellow color...) NOTE: to keep dimensions matched, convolution gate instead of regular sigmoid
        # gate (T in paper)
        gate = Conv2D(size_list[i_hidden + config.CNN_ResBlock_conv_per_block - 1], kernel_size=filt_list[-1],
                      padding='same', activation='sigmoid',
                      bias_initializer=tensorflow.keras.initializers.Constant(gate_bias))(value)
        # negated (C in paper)
        negated_gate = Lambda(lambda x: 1.0 - x, output_shape=(size_list[-1],))(gate)
        # use ResBlock as the Transformation
        transformed = ResBlock(x=value)
        transformed_gated = Multiply()([gate, transformed])
        # UpSample value if needed
        if value.shape.as_list()[-1] != negated_gate.shape.as_list()[-1]:
            r = negated_gate.shape.as_list()[-1] / value.shape.as_list()[-1]
            assert not (bool(r % 1))
            value = tf.keras.layers.UpSampling3D(size=(1, 1, int(r)))(value)
        identity_gated = Multiply()([negated_gate, value])
        value = Add()([transformed_gated, identity_gated])
        return value

    def skip_connection_layer(value):
        nonlocal i_hidden
        # use ResBlock as the Transformation
        transformed = ResBlock(x=value)
        if value.shape.as_list()[-1] != transformed.shape.as_list()[-1]:
            r = transformed.shape.as_list()[-1] / value.shape.as_list()[-1]
            assert not (bool(r % 1))
            # apply convolution as transformation
            value = Conv2D(size_list[i_hidden - 1], kernel_size=filt_list[i_hidden - 1], padding='same')(value)
        value = Add()([value, transformed])
        return value

    def ResBlock(x):
        for i in range(config.CNN_ResBlock_conv_per_block):
            nonlocal i_hidden  # to keep i_hidden "global" to all functions under CNNResBlockModel()
            lamda_cnn = 0.0 if config.use_l2_in_cnn is False else lamda
            x = Conv2D(size_list[i_hidden], kernel_size=filt_list[i_hidden], padding='same',
                       bias_regularizer=regularization(lamda_cnn),
                       kernel_regularizer=regularization(lamda_cnn),
                       kernel_initializer=kernel_initalizer)(x)
            x = activation(activation_name, x)
            if config.use_batch_norm is True:
                x = BatchNormalization()(x)
            i_hidden = i_hidden + 1
        return x

    def ResBlockLane(x):
        nonlocal i_hidden
        # ResBlocks
        for i in range(len(config.CNN_ResBlock_highway)):
            if config.CNN_ResBlock_highway[i] == "Highway":
                x = highway_layer(value=x)
            elif config.CNN_ResBlock_highway[i] == "Skip":
                x = skip_connection_layer(value=x)
            elif config.CNN_ResBlock_highway[i] == "None":
                x = ResBlock(x=x)
            else:
                raise Exception('only Highway/Skip/None is allowed !')
            # MaxPool and Dropout
            if config.CNN_ResBlock_dropout[i] != 0:
                x = Dropout(rate=config.CNN_ResBlock_dropout[i])(x)
            x = MaxPooling2D(pool_size=pool_list[i])(x)
        return x

    global background_implicit_inference
    # parameters
    kernel_initalizer = config.kernel_initalizer  # default is 'glorot_uniform'
    lamda = config.Regularization_term
    p_dropout = config.dropout
    activation_name = config.activation
    filt_dim2_list = config.Filter_shape_dim1 if config.Filter_shape_symmetric else config.Filter_shape_dim2
    filt_list = [(x, y) for x, y in zip(config.Filter_shape_dim1, filt_dim2_list)]
    pool_list = [(x, y) for x, y in zip(config.Pool_shape_dim1, config.Pool_shape_dim2)]
    size_list = config.hidden_size
    dense_list = config.Dense_size
    input_shape = config.model_input_dim
    p_dropout_conv1d = config.CNN_ResBlock_dropout_conv1d
    p_dropout_after_all_conv2d = config.dropout_after_all_conv2d
    p_dropout_dense = config.Dense_dropout
    i_hidden = 0

    # Input Layer
    input_layer = Input(shape=input_shape)
    assert len(size_list) == len(filt_list)
    assert len(pool_list) == len(config.CNN_ResBlock_highway) == len(config.CNN_ResBlock_dropout)
    assert config.CNN_ResBlock_conv_per_block * len(config.CNN_ResBlock_highway) == len(size_list)
    assert len(config.Conv1D_size) == len(config.Conv1D_kernel)

    if config.ResBlockDouble:
        x1 = input_layer
        x1 = ResBlockLane(x1)
        i_hidden = 0 # zero the hidden sizes counter
        x2 = input_layer
        x2 = ResBlockLane(x2)
        x = Add()([x1, x2])
    else:
        x = input_layer
        # ResBlocks
        x = ResBlockLane(x)
    # Flatten
    x = Flatten()(x)
    if p_dropout_after_all_conv2d != 0:
        x = Dropout(rate=p_dropout_after_all_conv2d)(x)
    # Conv1D
    if len(config.Conv1D_size) != 0:
        x = tf.expand_dims(x, axis=-1)
    for i in range(len(config.Conv1D_size)):
        x = Conv1D(filters=config.Conv1D_size[i], kernel_size=config.Conv1D_kernel[i],
                   kernel_initializer=kernel_initalizer)(x)
        x = activation(activation_name, x)
        if config.use_batch_norm is True:
            x = BatchNormalization()(x)
        if p_dropout_conv1d[i] != 0.0:
            x = Dropout(rate=p_dropout_conv1d[1])(x)
    # post-Conv1D
    if len(config.Conv1D_size) != 0:
        x = MaxPooling1D(pool_size=config.Conv1D_pool)(x)
        # x = BatchNormalization()(x)
        x = Flatten()(x)

    # Dense
    for i in range(len(dense_list)):
        x = Dense(dense_list[i], kernel_regularizer=regularization(lamda), kernel_initializer=kernel_initalizer)(x)
        x = activation(activation_name, x)
        if config.use_batch_norm is True:
            x = BatchNormalization()(x)
        if p_dropout_dense[i] != 0.0:
            x = Dropout(rate=p_dropout_dense[i])(x)
    # x = Dropout(rate=p_dropout)(x)
    # x = BatchNormalization()(x)
    if config.learn_background:
        x = Dense(3, activation='softmax')(x)
    else:
        x = Dense(1, activation='sigmoid')(x)
    output_layer = x
    model = Model(input_layer, output_layer)
    if config.learn_background:
        if config.background_implicit_inference:
            background_implicit_inference = True
        model = BlockBackgroundModel(input_layer, output_layer)
    # else:
    #     model = Model(input_layer, output_layer)
    # model.summary()
    return model


def CNN3D_Model(config):
    lamda = config.Regularization_term
    filt1 = (config.Filter_shape_dim1[0], config.Filter_shape_dim2[0], config.Filter_shape_dim3[0])
    filt2 = (config.Filter_shape_dim1[1], config.Filter_shape_dim2[1], config.Filter_shape_dim3[1])
    size1 = config.hidden_size[0]
    size2 = config.hidden_size[1]
    pool_shape1 = (config.Pool_shape_dim1[0], config.Pool_shape_dim2[0], config.Pool_shape_dim3[0])
    pool_shape2 = (config.Pool_shape_dim1[1], config.Pool_shape_dim2[1], config.Pool_shape_dim3[1])
    input_shape = config.model_input_dim
    dense_size1 = config.Dense_size[0]
    dense_size2 = config.Dense_size[1]
    p_dropout = config.dropout

    model = Sequential()
    model.add(Conv3D(size1, kernel_size=filt1, activation='relu', bias_regularizer='l2',
                     input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=pool_shape1))
    model.add(Conv3D(size2, kernel_size=filt2, activation='relu', bias_regularizer='l2'))
    model.add(MaxPooling3D(pool_size=pool_shape2))
    model.add(Flatten())
    model.add(Dense(dense_size1, kernel_regularizer='l2', activation='relu'))
    model.add(Dense(dense_size2, kernel_regularizer='l2', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def VGG16Model(config):
    ssl._create_default_https_context = ssl._create_unverified_context

    vgg_model = VGG16(weights='imagenet', include_top=False)
    vgg_model.trainable = False

    input_tensor = Input(shape=(config.model_input_dim[0], config.model_input_dim[1], config.model_input_dim[2]),
                         name='image_input')

    x = Conv2D(3, (3, 3), padding='same')(input_tensor)  # x has a dimension of (IMG_SIZE,IMG_SIZE,3)
    out = vgg_model(x)

    x = Flatten(name='flatten')(out)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dense(1, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=input_tensor, outputs=x)

    return model


def VGG19Model(config):
    ssl._create_default_https_context = ssl._create_unverified_context

    vgg_model = VGG19(weights='imagenet', include_top=False)
    vgg_model.trainable = False

    input_tensor = Input(shape=(config.model_input_dim[0], config.model_input_dim[1], config.model_input_dim[2]),
                         name='image_input')

    x = Conv2D(3, (3, 3), padding='same')(input_tensor)  # x has a dimension of (IMG_SIZE,IMG_SIZE,3)
    out = vgg_model(x)

    x = Flatten(name='flatten')(out)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dense(1, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=input_tensor, outputs=x)

    return model


def Resnet50Model(config):
    ssl._create_default_https_context = ssl._create_unverified_context

    resnet_model = ResNet50(weights='imagenet', include_top=False)

    # Set all layers trainable to False (except final conv layer)
    for layer in resnet_model.layers:
        layer.trainable = False

    for layer_index in range(150, 175):
        resnet_model.layers[layer_index].trainable = True

    input_tensor = Input(shape=(config.model_input_dim[0], config.model_input_dim[1], config.model_input_dim[2]),
                         name='image_input')

    x = Conv2D(3, (3, 3), padding='same')(input_tensor)  # x has a dimension of (IMG_SIZE,IMG_SIZE,3)
    out = resnet_model(x)

    x = Flatten(name='flatten')(out)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dense(1, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=input_tensor, outputs=x)

    return model


def CNNModel(config):
    input_shape = config.model_input_dim
    lamda = config.Regularization_term
    p_dropout = config.dropout
    filt1 = (config.Filter_shape_dim1[0], config.Filter_shape_dim2[0])
    filt2 = (config.Filter_shape_dim1[1], config.Filter_shape_dim2[1])
    filt3 = (config.Filter_shape_dim1[2], config.Filter_shape_dim2[2])

    size1 = config.hidden_size[0]
    size2 = config.hidden_size[1]
    size3 = config.hidden_size[2]

    pool_shape1 = (config.Pool_shape_dim1[0], config.Pool_shape_dim2[0])
    pool_shape2 = (config.Pool_shape_dim1[1], config.Pool_shape_dim2[1])
    pool_shape3 = (config.Pool_shape_dim1[2], config.Pool_shape_dim2[2])

    dense_size1 = config.Dense_size[0]
    dense_size2 = config.Dense_size[1]
    dense_size3 = config.Dense_size[2]
    model = Sequential()
    model.add(Conv2D(size1, kernel_size=filt1, activation='relu', bias_regularizer=keras.regularizers.l2(lamda),
                     kernel_regularizer=keras.regularizers.l2(lamda), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=pool_shape1))
    if p_dropout != 0:
        model.add(Dropout(rate=p_dropout))
    model.add(Conv2D(size2, kernel_size=filt2, activation='relu', bias_regularizer=keras.regularizers.l2(lamda),
                     kernel_regularizer=keras.regularizers.l2(lamda)))
    model.add(MaxPooling2D(pool_size=pool_shape2))
    if p_dropout != 0:
        model.add(Dropout(rate=p_dropout))
    model.add(Conv2D(size3, kernel_size=filt3, activation='relu', bias_regularizer=keras.regularizers.l2(lamda),
                     kernel_regularizer=keras.regularizers.l2(lamda)))
    model.add(MaxPooling2D(pool_size=pool_shape3))
    if p_dropout != 0:
        model.add(Dropout(rate=p_dropout * 2))
    model.add(Flatten())
    model.add(Dense(dense_size1, kernel_regularizer=keras.regularizers.l2(lamda), activation='relu'))
    model.add(Dense(dense_size2, kernel_regularizer=keras.regularizers.l2(lamda), activation='relu'))
    model.add(Dense(dense_size3, kernel_regularizer=keras.regularizers.l2(lamda), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def RESNETModel(config):
    # baseModel = ResNet50(weights=None, include_top=False,
    #                      input_tensor=Input(shape=config.model_input_dim))
    # # construct the head of the model that will be placed on top of the
    # # the base model
    # headModel = baseModel.output
    # headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
    # headModel = Flatten(name="flatten")(headModel)
    # headModel = Dense(256, activation="relu")(headModel)
    # headModel = Dropout(0.5)(headModel)
    # headModel = Dense(1, activation="sigmoid")(headModel)
    # # place the head FC model on top of the base model (this will become
    # # the actual model we will train)
    # model = Model(inputs=baseModel.input, outputs=headModel)
    # # loop over all layers in the base model and freeze them so they will
    # # *not* be updated during the training process
    # for layer in baseModel.layers:
    #     layer.trainable = False
    # model.summary()
    # return model
    if config.resnet_depth == 18:
        return res_net.ResnetBuilder().build_resnet_18(input_shape=config.model_input_dim, num_outputs=1)
    elif config.resnet_depth == 34:
        return res_net.ResnetBuilder().build_resnet_34(input_shape=config.model_input_dim, num_outputs=1)
    elif config.resnet_depth == 50:
        return res_net.ResnetBuilder().build_resnet_50(input_shape=config.model_input_dim, num_outputs=1)
    elif config.resnet_depth == 101:
        return res_net.ResnetBuilder().build_resnet_101(input_shape=config.model_input_dim, num_outputs=1)
    elif config.resnet_depth == 152:
        return res_net.ResnetBuilder().build_resnet_152(input_shape=config.model_input_dim, num_outputs=1)
    else:
        return None


def LSTMModel(config):
    def LSTM_layer(x, config, return_sequences):
        x = Bidirectional(
            LSTM(config.lstm_size, return_sequences=return_sequences, recurrent_regularizer=l2(lamda),
                 activity_regularizer=l2(lamda),
                 bias_regularizer=l2(lamda), kernel_regularizer=l2(lamda), activation=None), input_shape=input_shape)(x)
        if config.lstm_use_batch_norm:
            x = BatchNormalization(x)
        if config.lstm_use_layer_norm:
            x = LayerNormalization(x)
        x = Activation('tanh')(x)
        if config.lstm_dropout_rate > 0:
            x = SpatialDropout1D(rate=config.lstm_dropout_rate)(x)
        return x

    # parameters
    lamda = config.lstm_l2_reg
    bidir = config.lstm_use_bidir
    input_shape = (config.model_input_dim[0], config.model_input_dim[1])

    # input layer
    input_layer = Input(shape=input_shape)
    x = input_layer
    if config.lstm_dropout_rate > 0:
        x = SpatialDropout1D(rate=config.lstm_dropout_rate)(x)
    # encoder
    x = LSTM_layer(x, config, return_sequences=True)
    # decoder
    x = LSTM_layer(x, config, return_sequences=False)
    # classification
    x = Dense(1, activation='sigmoid')(x)  # changed to 1, binary classification
    output_layer = x
    model = Model(input_layer, output_layer)
    # model.summary()

    return model


def CNNResBlockModelComplex(config):
    def activation(activation_name, x):
        if activation_name == 'leaky_relu':
            return LeakyReLU(alpha=config.alpha)(x)
        else:
            return Activation(activation_name)(x)

    def highway_layer(value, gate_bias=-3):
        # https://towardsdatascience.com/review-highway-networks-gating-function-to-highway-image-classification-5a33833797b5
        nonlocal i_hidden  # to keep i_hidden "global" to all functions under CNNResBlockModel()
        dim = K.int_shape(value)[-1]
        # gate_bias_initializer = tensorflow.keras.initializers.Constant(gate_bias)
        # gate = Dense(units=dim, bias_initializer=gate_bias_initializer)(value)
        # gate = Activation("sigmoid")(gate)
        # TODO (just for yellow color...) NOTE: to keep dimensions matched, convolution gate instead of regular sigmoid
        # gate (T in paper)
        gate = Conv2D(size_list[i_hidden + config.CNN_ResBlock_conv_per_block - 1], kernel_size=filt_list[-1],
                      padding='same', activation='sigmoid',
                      bias_initializer=tensorflow.keras.initializers.Constant(gate_bias))(value)
        # negated (C in paper)
        negated_gate = Lambda(lambda x: 1.0 - x, output_shape=(size_list[-1],))(gate)
        # use ResBlock as the Transformation
        transformed = ResBlock(x=value)
        transformed_gated = Multiply()([gate, transformed])
        # UpSample value if needed
        if value.shape.as_list()[-1] != negated_gate.shape.as_list()[-1]:
            r = negated_gate.shape.as_list()[-1] / value.shape.as_list()[-1]
            assert not (bool(r % 1))
            value = tf.keras.layers.UpSampling3D(size=(1, 1, int(r)))(value)
        identity_gated = Multiply()([negated_gate, value])
        value = Add()([transformed_gated, identity_gated])
        return value

    def ResBlock(x):
        for i in range(config.CNN_ResBlock_conv_per_block):
            nonlocal i_hidden  # to keep i_hidden "global" to all functions under CNNResBlockModel()
            lamda_cnn = 0.0 if config.use_l2_in_cnn is False else lamda
            x = Conv2D(size_list[i_hidden], kernel_size=filt_list[i_hidden], padding='same',
                       bias_regularizer=keras.regularizers.l2(lamda_cnn),
                       kernel_regularizer=keras.regularizers.l2(lamda_cnn))(x)
            x = activation(activation_name, x)
            x = BatchNormalization()(x)
            i_hidden = i_hidden + 1
        return x

    if config.with_iq_matrices is False:
        raise Exception('This model support only operation for IQ representation')
    global background_implicit_inference
    # parameters
    lamda = config.Regularization_term
    p_dropout = config.dropout
    activation_name = config.activation
    filt_dim2_list = config.Filter_shape_dim1 if config.Filter_shape_symmetric else config.Filter_shape_dim2
    filt_list = [(x, y) for x, y in zip(config.Filter_shape_dim1, filt_dim2_list)]
    pool_list = [(x, y) for x, y in zip(config.Pool_shape_dim1, config.Pool_shape_dim2)]
    size_list = config.hidden_size
    dense_list = config.Dense_size
    input_shape = config.model_input_dim
    p_dropout_conv1d = config.CNN_ResBlock_dropout_conv1d
    p_dropout_after_all_conv2d = config.dropout_after_all_conv2d
    i_hidden = 0

    # Input Layer
    input_layer = Input(shape=input_shape)
    assert len(size_list) == len(filt_list)
    assert len(pool_list) == len(config.CNN_ResBlock_highway) == len(config.CNN_ResBlock_dropout)
    assert config.CNN_ResBlock_conv_per_block * len(config.CNN_ResBlock_highway) == len(size_list)
    assert len(config.Conv1D_size) == len(config.Conv1D_kernel)

    x = input_layer
    real_part = tf.expand_dims(x[:, :, :, 0], axis=-1)
    imag_part = tf.expand_dims(x[:, :, :, 1], axis=-1)

    real_part_output = Conv2D(size_list[0], kernel_size=filt_list[0], padding='same')(real_part)
    imag_part_output = Conv2D(size_list[0], kernel_size=filt_list[0], padding='same')(imag_part)

    real = tf.expand_dims(real_part_output, axis=-1)
    imag = tf.expand_dims(imag_part_output, axis=-1)
    filter_output = tf.concat([real, imag], axis=-1)
    x = complex_activation()(filter_output)
    # ResBlocks
    for i in range(len(config.CNN_ResBlock_highway)):
        if config.CNN_ResBlock_highway[i]:
            # True = use Highway
            x = highway_layer(value=x)
        else:
            # False = don't use Highway
            x = ResBlock(x=x)
        # MaxPool and Dropout
        if config.CNN_ResBlock_dropout[i] != 0:
            x = Dropout(rate=config.CNN_ResBlock_dropout[i])(x)
        x = MaxPooling2D(pool_size=pool_list[i])(x)
    # Flatten
    x = Flatten()(x)

    # Conv1D
    if len(config.Conv1D_size) != 0:
        x = tf.expand_dims(x, axis=-1)
    for i in range(len(config.Conv1D_size)):
        x = Conv1D(filters=config.Conv1D_size[i], kernel_size=config.Conv1D_kernel[i])(x)
        x = activation(activation_name, x)
        x = BatchNormalization()(x)
        if p_dropout_conv1d[i] != 0.0:
            x = Dropout(rate=p_dropout_conv1d[1])(x)
    # post-Conv1D
    if len(config.Conv1D_size) != 0:
        x = MaxPooling1D(pool_size=config.Conv1D_pool)(x)
        # x = BatchNormalization()(x)
        x = Flatten()(x)

    # Dense
    for i in range(len(dense_list)):
        x = Dense(dense_list[i], kernel_regularizer=keras.regularizers.l2(lamda))(x)
        x = activation(activation_name, x)
        if p_dropout_after_all_conv2d != 0 and len(config.Conv1D_size) == 0:
            x = Dropout(rate=p_dropout_after_all_conv2d)(x)
        x = BatchNormalization()(x)
    x = Dropout(rate=p_dropout)(x)
    # x = BatchNormalization()(x)
    if config.learn_background:
        x = Dense(3, activation='softmax')(x)
    else:
        x = Dense(1, activation='sigmoid')(x)
    output_layer = x
    model = Model(input_layer, output_layer)
    if config.learn_background:
        if config.background_implicit_inference:
            background_implicit_inference = True
        model = BlockBackgroundModel(input_layer, output_layer)
    # else:
    #     model = Model(input_layer, output_layer)
    # model.summary()
    return model
