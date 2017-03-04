# coding=utf-8

import os
import time
import numpy as np
import DataManager as dm
import keras.backend as K
from keras.layers import Dense, Activation, GRU, Dropout, Merge
from keras.metrics import top_k_categorical_accuracy, precision
from keras.models import Sequential, save_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

dmr = dm.DataManager()
signature = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())



def top_k_class(y_true, y_pred, tk, pk):
    # chance of predict_class_highest in top k classes
    shap = y_pred.get_shape()
    lenth = shap[len(shap) - 1]
    if pk <= lenth and tk <= lenth:
        y_pred_k = y_pred[::, lenth - pk:lenth]
        y_true_k = y_true[::, lenth - tk:lenth]
        true_positives = K.sum(K.round(K.clip(y_true_k * y_pred_k, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred_k, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


def top_t1p1_class(y_true, y_pred):
    return top_k_class(y_true, y_pred, 1, 1)

def top_t2p1_class(y_true, y_pred):
    return top_k_class(y_true, y_pred, 2, 1)

def top_t4p1_class(y_true, y_pred):
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


def build_model(params):
    """
    The function builds a keras Sequential model
    :param lookback: number of previous time steps as int
    :param batch_size: batch_size as int, defaults to 1
    :return: keras Sequential model
    """

    print "[ build_model ]... with params" + str(params)
    lookback = params['lookback']
    batch_size = params['batch_size']
    input_dim = params['indim']
    output_dim = params['outdim']

    model = Sequential()
    model.add(GRU(64,
                  activation='tanh',
                  batch_input_shape=(batch_size, lookback, input_dim),
                  stateful=False,
                  return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                  metrics=['precision', eval(params['custmetric']), 'fmeasure'])
    print "Finish building model"
    return model


def build_model2(lookback, batch_size, input_dim, output_dim):
    """
    The function builds a keras Sequential model
    :param lookback: number of previous time steps as int
    :param batch_size: batch_size as int, defaults to 1
    :return: keras Sequential model
    """
    rrnmodel = Sequential()
    rrnmodel.add(GRU(64,
                     activation='tanh',
                     batch_input_shape=(batch_size, lookback, input_dim),
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

    final_model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                        metrics=['precision', 'categorical_accuracy', 'fmeasure'])
    return final_model


def predict(model, data_x, data_y=None, batch_size=128, model_name=None):
    """
    The function predicates the output of input data_x and save it to a file
    :param model: trained model to predict
    :param data_x: input data, the last column of the last dimension is the stock code
    :param batch_size: batch_size as int, Optional
    :param data_y: the real output for validation, Optional
    :param model_name: the file name for store the prediction output, default value is the current date
    :return: prediction output
    """
    print "[ predict ]... using model " + model_name
    if model.stateful:
        data_x = data_x[:len(data_x) / batch_size * batch_size]
        data_y = data_y[:len(data_y) / batch_size * batch_size]
    proba = model.predict_proba(data_x, verbose=0, batch_size=batch_size)
    if data_y is None:
        # output (prediction, stock code, open, close, high, low)
        out = np.column_stack([proba, data_x[:, 0, 0:3]])
    else:
        # output (prediction, stock code, real output)
        out = np.column_stack([proba, data_y])
    sortout = out[(-out[:, proba.shape[proba.ndim - 1] - 1]).argsort(), :]

    if model_name is None:
        filename = signature
    else:
        filename = model_name
    if not __debug__:
        np.savetxt("./models/" + filename + "/result.txt", sortout, fmt='%f')
    else:
        print sortout[0:200, 2:]
    return sortout
