# coding=utf-8

import keras.backend as K
from keras.layers import Dense, Activation, GRU, Dropout
from keras.models import Sequential, save_model, load_model
from keras.utils import np_utils
import numpy as np

import DataManager as dm
import Symbols
import time


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


def main():
    rebuild = True
    if __debug__ & rebuild:
        look_back = 5
        batch_size = 1
        epoch = 1
        stocks = 2
    else:
        look_back = 5
        batch_size = 256
        epoch = 100
        stocks = 1500

    dmr = dm.DataManager()
    symbols = Symbols.symbols

    dataset = dmr.create_dataset(symbols[0:stocks], look_back)
    train, test = dmr.split_dataset(dataset, 0.8, batch_size)
    train_x, train_y = dmr.split_label(train)
    test_x, test_y = dmr.split_label(test)

    out_y = test_y.copy()

    # can only target to pchange_price
    test_y = test_y[:,0]
    train_y = train_y[:,0]

    train_y[train_y < -2] = 11
    train_y[train_y < 2] = 12
    train_y[train_y < 11] = 13
    train_y = train_y - 11
    train_y = np_utils.to_categorical(train_y, 3)
    test_y[test_y < -2] = 11
    test_y[test_y < 2] = 12
    test_y[test_y < 11] = 13
    test_y = test_y - 11
    test_y = np_utils.to_categorical(test_y, 3)
    filename = None
    if rebuild:
        filename = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        model = build_model(look_back, batch_size, train_x.shape[2], train_y.shape[1])
        callback = model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=epoch, validation_data=(test_x, test_y))

        #Save models and metrics
        save_model(model, './models/' + filename+'.h5')
        save_model(model, './models/latest.h5')
        hist = dict(callback.history)
        for key in hist.keys():
            np.savetxt("./models/"+filename+"_"+key+".txt", hist[key])
    else:
        filename = 'latest'
        try:
            model = load_model('./models/latest.h5')
        except:
            raise "Can't load model at: ./models/latest.h5"

    # preds = model.predict_classes(test_x, batch_size=batch_size)
    proba = model.predict_proba(test_x, verbose=0, batch_size=batch_size)

    out = np.column_stack([proba,out_y])
    sortout = out[(-out[:,2]).argsort(), :]
    np.savetxt("./models/" + filename + "_result.txt", sortout, fmt='%f')
    print sortout[0:200][0:5]

    # train_predict = model.predict(train_x, batch_size)
    # test_predict = model.predict(test_x, batch_size)

if __name__ == '__main__':
    main()
