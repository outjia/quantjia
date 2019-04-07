# coding=utf-8

from keras.layers import Dense, Activation, GRU, Dropout
from keras.models import Sequential


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


def build_model3(params):
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
                  batch_input_shape=(batch_size, lookback*knum,input_dim),
                  return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                  metrics=params['metrics'])
    print ("Finish building model")
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
