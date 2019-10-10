# coding=utf-8

from keras.layers import Dense, Activation, GRU, Dropout, Conv2D, Flatten
from keras.models import Sequential
import keras.backend as K


def nbuild_rmodel(params):
    """
    The function builds a keras Sequential model
    :param lookback: number of previous time steps as int
    :param batch_size: batch_size as int, defaults to 1
    :return: keras Sequential model
    """

    print ("[ build_model ]... with params:" + str(params))
    lookback = params['lookback']
    batch_size = params['batch_size']
    input_dim = params['indim']
    output_dim = params['outdim']
    knum = 240/int(params['ktype'])

    model = Sequential()
    model.add(GRU(16,
                  activation='tanh',
                  batch_input_shape=(batch_size, lookback*knum,input_dim),
                  return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                  metrics=params['metrics'])
    print ("Finish building model")
    return model


def nbuild_lrmodel(params):
    print ("[ build_model ]... with params" + str(params))
    lookback = params['lookback']
    batch_size = params['batch_size']
    input_dim = params['indim']
    output_dim = params['outdim']
    knum = 240/int(params['ktype'])

    model = Sequential()
    model.add(GRU(32,
                  activation='tanh',
                  batch_input_shape=(batch_size, lookback*knum,input_dim),
                  return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('tanh'))
    model.compile(loss='mse', optimizer='rmsprop')
    print ("Finish building model")
    return model


def build_rmodel(params):
    print ("[ build_model ]... with params" + str(params))
    lookback = params['lookback']
    batch_size = params['batch_size']
    input_dim = params['indim']
    output_dim = params['outdim']

    model = Sequential()
    model.add(GRU(64,
                  activation='tanh',
                  batch_input_shape=(batch_size, lookback, input_dim),
                  stateful=True,
                  return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                  metrics=params['metrics'])
    print ("Finish building model")
    return model


def build_cmodel(params):
    K.set_image_dim_ordering('th')
    print "[ build_model ]... with params" + str(params)
    channels = params['indim']
    output_dim = params['outdim']
    cols = 240/int(params['ktype'])
    rows = params['lookback']

    model = Sequential()
    model.add(Conv2D(8,(2,2), strides=(2, 2),input_shape=(rows, cols, channels),data_format = 'channels_last'))
    # model.add(AveragePooling2D(pool_size=2, strides=2))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    # model.add(Conv2D(8,(2,2),data_format='channels_last',padding="same"))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(4, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    # sdg = SGD(lr=0.01, decay=1e-6, momentum=0.8, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='sgd',
                  metrics=params['metrics'])
    print "Finish building model"
    return model


def build_model5(params):
    """
    The function builds a keras Sequential model
    :param lookback: number of previous time steps as int
    :param batch_size: batch_size as int, defaults to 1
    :return: keras Sequential model
    """

    print ("[ build_model ]... with params" + str(params))
    lookback = params['lookback']
    batch_size = params['batch_size']
    input_dim = params['indim']
    output_dim = params['outdim']

    model = Sequential()
    model.add(GRU(64,
                  activation='tanh',
                  batch_input_shape=(batch_size, lookback, input_dim),
                  stateful=True,
                  return_sequences=True))
    model.add(GRU(32,
                  activation='tanh',
                  batch_input_shape=(batch_size, lookback, 64),
                  stateful=True,
                  return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                  metrics=params['metrics'])
    print ("Finish building model")
    return model


if __name__ == "__main__":
    pass
