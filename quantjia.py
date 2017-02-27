# coding=utf-8

import time
import numpy as np
import DataManager as dm
import ModelManager as mdm
import Symbols
from keras.models import load_model

dmr = dm.DataManager()
signature = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
symbols = Symbols.symbols
rebuild = True
import sys

if __debug__:
    params = {
        'model_name':'Model_3',
        'look_back':5,
        'batch_size':256,
        'epoch':5,
        'stocks':3,
        'indim':0,
        'outdim':3
    }
else:
    params1 = {
        'model_name':'Model_4_l5_C3',
        'look_back':5,
        'batch_size':256,
        'epoch':100,
        'stocks':2000,
        'indim':0,
        'outdim':3
    }

    params3 = {
        'model_name':'Model_4_l22_C3',
        'look_back':22,
        'batch_size':256,
        'epoch':100,
        'stocks':2000,
        'indim':0,
        'outdim':3
    }
    params2 = {
        'model_name':'Model_4_l5_C10',
        'look_back':5,
        'batch_size':256,
        'epoch':100,
        'stocks':2000,
        'indim':0,
        'outdim':10
    }


def main():
    global  params
    params = params2
    dataset = dmr.create_dataset2(symbols[0:params['stocks']+1], params['look_back'])
    train, test = dmr.split_dataset(dataset, 0.7, params['batch_size'])
    bsdata_train, tsdata_train, rtdata_train, lbdata_train = dmr.split_label2(train)
    bsdata_test, tsdata_test, rtdata_test, lbdata_test = dmr.split_label2(test)
    out_y = lbdata_test.copy()

    # train_x = np.hstack([bsdata_train, tsdata_train])
    # test_x = np.hstack([bsdata_test,tsdata_test])
    train_x = tsdata_train
    test_x = tsdata_test
    params['indim'] = train_x.shape[train_x.ndim-1]
    # target to pchange_price
    # train_y = dmr.catnorm_data(lbdata_train[:, 1])
    # test_y = dmr.catnorm_data(lbdata_test[:, 1])
    # if rebuild:
    #     model = mdm.build_model4(params)
    #     mdm.train_model(model, params, train_x, train_y, test_x, test_y)
    # else:
    #     try:
    #         model = load_model('./model.h5')
    #     except:
    #         raise "Can't load model at: ./model.h5"
    # mdm.predict(model, test_x, out_y, params['batch_size'], params['model_name'])

    # target to pchange_price
    # train_y = dmr.catnorm_data10(lbdata_train[:, 1])
    # test_y = dmr.catnorm_data10(lbdata_test[:, 1])
    # model = mdm.build_model5(params)
    # mdm.train_model(model, params, train_x, train_y, test_x, test_y)
    # mdm.predict(model, test_x, out_y, params['batch_size'], params['model_name'])
    #
    # params = params2
    # model = mdm.build_model4(params)
    # mdm.train_model(model, params, train_x, train_y, test_x, test_y)
    # mdm.predict(model, test_x, out_y, params['batch_size'], params['model_name'])

    params = params1
    params['indim'] = train_x.shape[train_x.ndim - 1]
    # target to pchange_price
    train_y = dmr.catnorm_data(rtdata_train[:, 1])
    test_y = dmr.catnorm_data(rtdata_test[:, 1])
    model = mdm.build_model4(params)
    mdm.train_model(model, params, train_x, train_y, test_x, test_y)
    mdm.predict(model, test_x, rtdata_test, params['batch_size'], params['model_name'])

    # todaydata = dmr.get_todaydata(look_back=5, refresh=False)
    # mdm.predict_today(model, todaydata, batch_size)
    return

# def main2():
#     # dmr.refresh_data(data_type='BH', trytimes=10)
#     dataset = dmr.create_dataset(symbols[0:stocks], look_back)
#     train, test = dmr.split_dataset(dataset, 0.7, batch_size)
#     train_x, train_y = dmr.split_label(train)
#     test_x, test_y = dmr.split_label(test)
#     out_y = test_y.copy()
#     # target to pchange_price
#     train_y = dmr.catnorm_data(train_y[:, 0])
#     test_y = dmr.catnorm_data(test_y[:, 0])
#     if rebuild:
#         model = mdm.build_model(look_back, batch_size, train_x.shape[2], train_y.shape[1])
#         callback = model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=epoch, validation_data=(test_x, test_y))
#         if not __debug__:
#             #Save models and metrics
#             save_model(model, './models/' + signature+'.h5')
#             # save_model(model, './models/latest.h5')
#             hist = dict(callback.history)
#             for key in hist.keys():
#                 np.savetxt("./models/"+signature+"_"+key+".txt", hist[key])
#     else:
#         try:
#             model = load_model('./models/latest.h5')
#         except:
#             raise "Can't load model at: ./models/latest.h5"
#
#     mdm.predict_d(model, test_x, out_y, batch_size)
#
#     # todaydata = dmr.get_todaydata(look_back=5, refresh=False)
#     # mdm.predict_today(model, todaydata, batch_size)


def Model_6_l5_C3():
    params = {
        'model_name':'Model_6_l5_C3',
        'look_back':5,
        'batch_size':128,
        'epoch':60,
        'stocks':500,
        'indim':0,
        'outdim':3
    }
    dataset = dmr.create_dataset2(symbols[0:params['stocks']+1], params['look_back'])
    train, test = dmr.split_dataset(dataset, 0.7, params['batch_size'])
    bsdata_train, tsdata_train, rtdata_train, lbdata_train = dmr.split_label2(train)
    bsdata_test, tsdata_test, rtdata_test, lbdata_test = dmr.split_label2(test)
    train_x = tsdata_train
    test_x = tsdata_test
    train_y = dmr.catnorm_data(rtdata_train[:, 1])
    test_y = dmr.catnorm_data(rtdata_test[:, 1])
    params['indim'] = train_x.shape[train_x.ndim - 1]
    model = mdm.build_model4(params)
    mdm.train_model(model, params, train_x, train_y, test_x, test_y)
    mdm.predict(model, test_x, rtdata_test, params['batch_size'], params['model_name'])

    # todaydata = dmr.get_todaydata(look_back=5, refresh=False)
    # mdm.predict_today(model, todaydata, batch_size)
    return

def Model_6_l22_C3():
    params = {
        'model_name':'Model_6_l5_C3',
        'look_back':22,
        'batch_size':128,
        'epoch':60,
        'stocks':500,
        'indim':0,
        'outdim':3
    }
    dataset = dmr.create_dataset2(symbols[0:params['stocks']+1], params['look_back'])
    train, test = dmr.split_dataset(dataset, 0.75, params['batch_size'])
    bsdata_train, tsdata_train, rtdata_train, lbdata_train = dmr.split_label2(train)
    bsdata_test, tsdata_test, rtdata_test, lbdata_test = dmr.split_label2(test)
    train_x = tsdata_train
    test_x = tsdata_test
    train_y = dmr.catnorm_data(rtdata_train[:, 1])
    test_y = dmr.catnorm_data(rtdata_test[:, 1])
    params['indim'] = train_x.shape[train_x.ndim - 1]
    model = mdm.build_model4(params)
    mdm.train_model(model, params, train_x, train_y, test_x, test_y)
    mdm.predict(model, test_x, rtdata_test, params['batch_size'], params['model_name'])

    # todaydata = dmr.get_todaydata(look_back=5, refresh=False)
    # mdm.predict_today(model, todaydata, batch_size)
    return


def Model_4_l22_C2():
    params = {
        'model_name':'Model_4_l22_C2',
        'look_back':22,
        'batch_size':128,
        'epoch':60,
        'stocks':50,
        'indim':0,
        'outdim':2,
        'cat_func':dmr.catnorm_data2test
    }
    dataset = dmr.create_dataset2(symbols[0:params['stocks']+1], params['look_back'])
    train, test = dmr.split_dataset(dataset, 0.75, params['batch_size'])
    bsdata_train, tsdata_train, rtdata_train, lbdata_train = dmr.split_label2(train)
    bsdata_test, tsdata_test, rtdata_test, lbdata_test = dmr.split_label2(test)
    train_x = tsdata_train
    test_x = tsdata_test
    train_y = params['cat_func'](rtdata_train[:, 1])
    test_y = params['cat_func'](rtdata_test[:, 1])
    params['indim'] = train_x.shape[train_x.ndim - 1]
    model = mdm.build_model4(params)
    mdm.train_model(model, params, train_x, train_y, test_x, test_y)
    mdm.predict(model, test_x, rtdata_test, params['batch_size'], params['model_name'])
    mdm.predict(model, train_x, rtdata_train, params['batch_size'], params['model_name']+'_train')

    # todaydata = dmr.get_todaydata(look_back=5, refresh=False)
    # mdm.predict_today(model, todaydata, batch_size)
    return

def M1L5C3():
    params = {
        'model_name': 'M1L5C3_2',
        'look_back': 5,
        'batch_size': 256,
        'epoch': 120,
        'stocks': 100,
        'indim': 0,
        'outdim': 3,
        'cat_func': dmr.catnorm_data
    }
    dataset = dmr.create_dataset2(symbols[0:params['stocks'] + 1], params['look_back'])
    train, test = dmr.split_dataset(dataset, 0.75, params['batch_size'])
    bsdata_train, tsdata_train, rtdata_train, lbdata_train = dmr.split_label2(train)
    bsdata_test, tsdata_test, rtdata_test, lbdata_test = dmr.split_label2(test)
    train_x = tsdata_train
    test_x = tsdata_test
    train_y = params['cat_func'](rtdata_train[:, 1])
    test_y = params['cat_func'](rtdata_test[:, 1])
    params['indim'] = train_x.shape[train_x.ndim - 1]
    model = mdm.build_model(params)
    mdm.train_model(model, params, train_x, train_y, test_x, test_y)
    mdm.predict(model, test_x, rtdata_test, params['batch_size'], params['model_name'])
    mdm.predict(model, train_x, rtdata_train, params['batch_size'], params['model_name']+'_train')

    # mdm.predict_d(model, test_x, rtdata_test, params['batch_size'])



def predict_today():
    try:
        model = load_model('./models/latest.h5')
    except:
        raise "Can't load model at: ./models/latest.h5"
    todaydata = dmr.get_todaydata(look_back=5, refresh=False)
    mdm.predict(model, todaydata, 256)


def _main_():
    model_map = {
        'Model_6_l5_C3': Model_6_l5_C3,
        'Model_6_l22_C3': Model_6_l22_C3,
        'Model_4_l22_C2': Model_4_l22_C2,
        'M1L5C3': M1L5C3,

    }
    for i in range(1, len(sys.argv)):
        if sys.argv[i] in model_map.keys():
            model_map[sys.argv[i]]()

if __name__ == '__main__':
    # Model_4_l22_C2()
    # dmr.refresh_data(trytimes=10)
    # main_old()
    _main_()

