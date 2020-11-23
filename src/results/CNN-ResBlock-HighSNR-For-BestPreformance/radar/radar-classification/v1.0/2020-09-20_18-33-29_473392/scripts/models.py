from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, Softmax, BatchNormalization, Dropout, LSTM, AveragePooling2D, Input
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from models import res_net


def build_model(config):
    if config.model_name == "radar-cnn":
        model = {'train': CNN_BNModel(config),
                 'eval': CNN_BNModel(config)}
    elif config.model_name == "cnn-baseline":
        model = {'train': BaselineCNNModel(config),
                 'eval': BaselineCNNModel(config)}
    elif config.model_name == "radar-lstm":
        model = {'train': LSTMModel(config),
                 'eval': LSTMModel(config)}
    elif config.model_name == "radar-resnet":
        model = {'train': RESNETModel(config),
                 'eval': RESNETModel(config)}
    else:
        raise ValueError("'{}' is an invalid model name")

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
    model.add(Conv2D(size1, kernel_size=filt1, activation='relu', bias_regularizer='l2',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=pool_shape1))
    model.add(Conv2D(size2, kernel_size=filt2, activation='relu', bias_regularizer='l2'))
    model.add(MaxPooling2D(pool_size=pool_shape2))
    model.add(Flatten())
    model.add(Dense(dense_size1, kernel_regularizer='l2', activation='relu'))
    model.add(Dense(dense_size2, kernel_regularizer='l2', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def CNNModel(config):
    model = Sequential()
    model.add(Conv2D(config.hidden_size[0], (3, 3), activation='relu', input_shape=config.model_input_dim))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(config.hidden_size[1], (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(config.model_output_dim, activation=None))
    model.add(Softmax(axis=-1))

    return model

def RESNETModel(config):
    baseModel = ResNet50(weights=None, include_top=False,
                         input_tensor=Input(shape=config.model_input_dim))
    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(256, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(1, activation="sigmoid")(headModel)
    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)
    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the training process
    for layer in baseModel.layers:
        layer.trainable = False
    model.summary()
    return model
    # return res_net.ResnetBuilder().build_resnet_18(input_shape=config.model_input_dim, num_outputs=1)

def LSTMModel(config):
    # build LSTM model
    size1 = config.hidden_size[0]
    lamda = config.Regularization_term
    # input_shape = (None, config.model_input_dim[-1])
    input_shape = (None, config.model_input_dim[1])
    # input_dim_list.insert(0, config.batch_size)
    # batch_input_shape = tuple(input_dim_list)
    p_dropout = config.dropout
    size2 = config.Dense_size[0]
    model_lstm = keras.Sequential()
    if lamda is None:
        model_lstm.add(
            LSTM(size1, activation='tanh', input_shape=input_shape))
    else:
        model_lstm.add(
            LSTM(size1, activation='tanh', input_shape=input_shape,
                 kernel_regularizer=keras.regularizers.l2(lamda),
                 bias_regularizer=keras.regularizers.l2(lamda),
                 recurrent_regularizer=keras.regularizers.l2(lamda)))
    if config.use_bn is True:
        model_lstm.add(BatchNormalization())
    if p_dropout is not None:
        model_lstm.add(Dropout(p_dropout))
    if size2 is not None:
        model_lstm.add(Dense(size2, activation='relu'))
    if p_dropout is not None:
        model_lstm.add(Dropout(p_dropout))
    model_lstm.add(Dense(1, activation='sigmoid'))
    model_lstm.summary()
    return model_lstm
