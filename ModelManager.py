# coding=utf-8

import os
import time
import numpy as np
from DataManager import *
import keras.backend as K
from keras.regularizers import l1, l1_l2, l2
from keras.layers import Dense, Activation, GRU, Dropout, Conv2D, LSTM,Flatten,MaxPool2D,AveragePooling2D
from keras.models import Sequential, save_model
from keras.optimizers import SGD

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


def nbuild_model(params):
    """
    The function builds a keras Sequential model
    :param lookback: number of previous time steps as int
    :param batch_size: batch_size as int, defaults to 1
    :return: keras Sequential model
    """
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


def nbuild_model2(params):
    """
    The function builds a keras Sequential model
    :param lookback: number of previous time steps as int
    :param batch_size: batch_size as int, defaults to 1
    :return: keras Sequential model
    """
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
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=params['metrics'])
    print "Finish building model"
    return model



if __name__ == "__main__":
    pass
