# coding=utf-8

import keras.backend as K
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, GRU, Dropout, Merge
from keras.models import Sequential, save_model, load_model
from keras.utils import np_utils
import numpy as np
import time
import DataManager as dm

signature = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
datapath = './data/new/'


def tpfn_metrics(y_true, y_pred):
    # print K.eval(y_true)
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)

    return {
        # 'size': K.sum(y_true, axis=1),
        # 'size_p': K.sum(y_pred, axis=1),
        'true_positive': tp,
        'false_positive': fp,
    }


def build_model(look_back, batch_size, input_dim, output_dim):
    """
    The function builds a keras Sequential model
    :param look_back: number of previous time steps as int
    :param batch_size: batch_size as int, defaults to 1
    :return: keras Sequential model
    """
    model = Sequential()
    model.add(GRU(64,
                  activation='tanh',
                  batch_input_shape=(batch_size, look_back, input_dim),
                  stateful=True,
                  return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['precision', 'categorical_accuracy', 'fmeasure'])
    return model

def build_model2(look_back, batch_size, input_dim, output_dim):
    """
    The function builds a keras Sequential model
    :param look_back: number of previous time steps as int
    :param batch_size: batch_size as int, defaults to 1
    :return: keras Sequential model
    """
    rrnmodel = Sequential()
    rrnmodel.add(GRU(64,
                  activation='tanh',
                  batch_input_shape=(batch_size, look_back, input_dim),
                  stateful=True,
                  return_sequences=False))
    rrnmodel.add(Dropout(0.3))
    rrnmodel.add(Dense(32, activation='tanh'))
    rrnmodel.add(Dropout(0.3))

    linearmodel = Sequential()
    linearmodel.add(Dense(8, input_dim=6))
    merged = Merge([rrnmodel, linearmodel], mode='concat')

    final_model = Sequential()
    final_model.add(merged)
    rrnmodel.add(Dense(16))
    final_model.add(Dense(3, activation='softmax'))

    final_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['precision', 'categorical_accuracy', 'fmeasure'])
    return final_model

def predict(model, data_x, data_y, batch_size):
    proba = model.predict_proba(data_x, verbose=0, batch_size=batch_size)
    out = np.column_stack([proba, data_y])
    sortout = out[(-out[:,2]).argsort(), :]
    print sortout[0:200, 0:5]
    if not __debug__: np.savetxt("./models/" + signature + "_result.txt", sortout, fmt='%f')
    return sortout

def predict_today(model, batch_size):
    #TODO merge predict and predict_today
    dmr = dm.DataManager()
    todaydata = dmr.get_todaydata(look_back=5, refresh=False)
    if todaydata is not None:
        if model.stateful:
            todaydata = todaydata[:len(todaydata) / batch_size * batch_size]
        proba = model.predict_proba(todaydata[:,:,0:todaydata.shape[2]-1], verbose=0, batch_size=batch_size)
    out = np.column_stack([proba, todaydata[:,0,todaydata.shape[2]-1], todaydata[:,0,0:3]])
    sortout = out[(-out[:,2]).argsort(), :]
    if not __debug__: np.savetxt("./models/" + signature + "_result.txt", sortout, fmt='%f')
    return sortout

