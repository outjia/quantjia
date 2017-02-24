# coding=utf-8

import keras.backend as K
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, GRU, Dropout
from keras.models import Sequential, save_model, load_model
from keras.utils import np_utils
import numpy as np

import DataManager as dm
import ModelManager as mdm
import Symbols
import time

dmr = dm.DataManager()
signature = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
symbols = Symbols.symbols
rebuild = False

if __debug__:
    look_back = 5
    batch_size = 256
    epoch = 100
    stocks = 1500
    # look_back = 5
    # batch_size = 1
    # epoch = 1
    # stocks = 2
else:
    look_back = 5
    batch_size = 256
    epoch = 100
    stocks = 1500

def main():
    if rebuild:
        dataset = dmr.create_dataset(symbols[0:stocks], look_back)
        train, test = dmr.split_dataset(dataset, 0.8, batch_size)
        train_x, train_y = dmr.split_label(train)
        test_x, test_y = dmr.split_label(test)
        out_y = test_y.copy()
        # target to pchange_price
        train_y = dmr.catnorm_data(train_y[:, 0])
        test_y = dmr.catnorm_data(test_y[:, 0])

        model = mdm.build_model(look_back, batch_size, train_x.shape[2], train_y.shape[1])
        callback = model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=epoch, validation_data=(test_x, test_y))
        if not __debug__:
            #Save models and metrics
            save_model(model, './models/' + signature+'.h5')
            save_model(model, './models/latest.h5')
            hist = dict(callback.history)
            for key in hist.keys():
                np.savetxt("./models/"+signature+"_"+key+".txt", hist[key])
    else:
        try:
            model = load_model('./models/latest.h5')
        except:
            raise "Can't load model at: ./models/latest.h5"

    # mdm.predict(test_x, out_y, batch_size)
    mdm.predict_today(model, batch_size)

if __name__ == '__main__':
    main()


def test_plot():
    d = np.loadtxt("./models/2017_02_23_18_23_20/2017_02_23_18_23_20_result.txt")
    dmr.plot_out(d, 2, 3)
