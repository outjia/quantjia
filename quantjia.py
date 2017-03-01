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
    pass
else:
    pass


def main():
    params = {
        'model_name':'Model_4_l5_C10',
        'look_back':5,
        'batch_size':256,
        'epoch':100,
        'stocks':2000,
        'indim':0,
        'outdim':10
    }
    return



def M6T5C3():
    params = {
        'model_name':'M6T5C3',
        'look_back':5,
        'batch_size':128,
        'epoch':60,
        'stocks':500,
        'indim':0,
        'outdim':3
    }
    dataset = dmr.create_dataset(symbols[0:params['stocks']+1], params['look_back'])
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


def M6T5C3():
    params = {
        'model_name':'M6T5C3',
        'look_back':22,
        'batch_size':128,
        'epoch':60,
        'stocks':500,
        'indim':0,
        'outdim':3
    }
    dataset = dmr.create_dataset(symbols[0:params['stocks']+1], params['look_back'])
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


def M4T22C2():
    params = {
        'model_name':'M4T22C2',
        'look_back':22,
        'batch_size':128,
        'epoch':60,
        'stocks':50,
        'indim':0,
        'outdim':2,
        'cat_func':dmr.catnorm_data2test
    }
    dataset = dmr.create_dataset(symbols[0:params['stocks']+1], params['look_back'])
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

def M1T5C3():
    params = {
        'model_name': 'M1T5C3_2',
        'look_back': 5,
        'batch_size': 256,
        'epoch': 120,
        'stocks': 100,
        'indim': 0,
        'outdim': 3,
        'cat_func': dmr.catnorm_data
    }
    dataset = dmr.create_dataset(symbols[0:params['stocks'] + 1], params['look_back'])
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


def M1T10C3():
    params = {
        'model_name': 'M1T5C3_2',
        'look_back': 5,
        'batch_size': 256,
        'epoch': 120,
        'stocks': 100,
        'indim': 0,
        'outdim': 3,
        'cat_func': dmr.catnorm_data
    }
    dataset = dmr.create_dataset(symbols[0:params['stocks'] + 1], params['look_back'])
    train, test = dmr.split_dataset(dataset, 0.75, params['batch_size'])
    bsdata_train, tsdata_train, rtdata_train, lbdata_train = dmr.split_label2(train)
    bsdata_test, tsdata_test, rtdata_test, lbdata_test = dmr.split_label2(test)

    opent1_train = rtdata_train[:,2].reshape(-1,1)
    lbdata_train = (lbdata_train[:,2:] - opent1_train)/opent1_train * 10

    opent1_test = rtdata_test[:,2].reshape(-1,1)
    lbdata_test = (lbdata_test[:,2:] - opent1_test)/opent1_test * 10

    train_x = tsdata_train[:,1]
    test_x = tsdata_test[:,1]
    train_y = params['cat_func'](lbdata_train[:, 2])
    test_y = params['cat_func'](lbdata_test[:, 2])
    params['indim'] = train_x.shape[train_x.ndim - 1]
    model = mdm.build_model(params)
    mdm.train_model(model, params, [train_x, ], train_y, test_x, test_y)
    mdm.predict(model, test_x, rtdata_test, params['batch_size'], params['model_name'])
    mdm.predict(model, train_x, rtdata_train, params['batch_size'], params['model_name']+'_train')


def M1T5C2():
    params = {
        'model_name': 'M1T5C2_D2',
        'look_back': 5,
        'batch_size': 256,
        'epoch': 100,
        'stocks': 300,
        'indim': 0,
        'outdim': 3,
        'cat_func': dmr.catnorm_data
    }
    # dmr.get_bsdata(True)
    dataset = dmr.create_dataset(symbols[0:params['stocks'] + 1], params['look_back'])
    train, test = dmr.split_dataset(dataset, 0.75, params['batch_size'],1488290043)
    bsdata_train, tsdata_train, rtdata_train, lbdata_train, tsdata_train_v, rtdata_train_v, lbdata_train_v = dmr.create_feeddata(train)
    bsdata_test, tsdata_test, rtdata_test, lbdata_test, tsdata_test_v, rtdata_test_v, lbdata_test_v = dmr.create_feeddata(test)

    # closed1_train = rtdata_train[:,-1].reshape(-1,1)
    # closed0_train = tsdata_train[:,-1,-1].reshape(-1,1)
    # train_y = (closed1_train - closed0_train)#/closed0_train * 100
    #
    # closed1_test = rtdata_test[:,-1].reshape(-1,1)
    # closed0_test = tsdata_test[:,-1,-1].reshape(-1,1)
    # test_y = (closed1_test - closed0_test)#/closed0_test * 100

    train_x = tsdata_train[:,:,1:]
    test_x = tsdata_test[:,:,1:]
    train_y = params['cat_func'](lbdata_train[:, -2])
    test_y = params['cat_func'](lbdata_test[:,-2])
    params['indim'] = train_x.shape[train_x.ndim - 1]
    # model = mdm.build_model(params)
    # mdm.train_model(model, params, train_x, train_y, test_x, test_y)
    model = load_model('./models/model.h5',custom_objects={'top_t1p1_class':mdm.top_t1p1_class})
    mdm.predict(model, test_x, np.hstack([rtdata_test_v,lbdata_test_v[:,2:]]), params['batch_size'], params['model_name'])

def MTODAY():
    try:
        model = load_model('./models/model.h5',custom_objects={'top_t1p1_class':mdm.top_t1p1_class})
    except:
        raise "Can't load model at: ./models/model.h5"
    todaydata = dmr.get_todaydata(look_back=5, refresh=False)
    mdm.predict(model, todaydata, 256)


def _main_():
    for i in range(1, len(sys.argv)):
        eval(sys.argv[i])()

if __name__ == '__main__':
    # _main_()
    M1T5C2()
