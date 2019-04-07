# coding=utf-8

import os
import time
import numpy as np
from DataManager import *
import keras.backend as K
from keras.regularizers import l1, l1_l2, l2
from keras.layers import Dense, Activation, GRU, Dropout, GRUCell
# from keras.metrics import
from keras.models import Sequential, save_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

signature = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def top_k_recall(y_true, y_pred, k):
    # chance of predict_class_highest in top k classes
    y_pred_k = y_pred[::, - k:]
    y_true_k = y_true[::, - k:]
    true_positives = K.sum(K.round(K.clip(y_true_k * y_pred_k, 0, 1)))
    positives = K.sum(K.round(K.clip(y_true_k, 0, 1)))
    recall = true_positives / (positives + K.epsilon())
    return recall


def top1_recall(y_true, y_pred):
    return top_k_recall(y_true, y_pred, 1)


def top2_recall(y_true, y_pred):
    return top_k_recall(y_true, y_pred, 2)


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def top_k_class(y_true, y_pred, tk, pk):
    # chance of predict_class_highest in top k classes
    y_pred_k = y_pred[::, - pk:]
    y_true_k = y_true[::, - tk:]
    true_positives = K.sum(K.round(K.clip(y_true_k * y_pred_k, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred_k, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def top_t1p1(y_true, y_pred):
    return top_k_class(y_true, y_pred, 1, 1)


def top_t2p1(y_true, y_pred):
    return top_k_class(y_true, y_pred, 2, 1)


def top_t4p1(y_true, y_pred):
    return top_k_class(y_true, y_pred, 4, 1)


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
