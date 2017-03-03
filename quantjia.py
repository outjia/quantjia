# coding=utf-8

import time
import datetime
import os
import numpy as np
import DataManager as dm
from DataManager import int2str
import ModelManager as mdm
import Symbols
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

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
        'lookback':5,
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
        'lookback':5,
        'batch_size':128,
        'epoch':60,
        'stocks':500,
        'indim':0,
        'outdim':3
    }
    dataset = dmr.create_dataset(symbols[0:params['stocks']+1], params['lookback'])
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

    # todaydata = dmr.get_todaydata(lookback=5, refresh=False)
    # mdm.predict_today(model, todaydata, batch_size)
    return


def M6T5C3():
    params = {
        'model_name':'M6T5C3',
        'lookback':22,
        'batch_size':128,
        'epoch':60,
        'stocks':500,
        'indim':0,
        'outdim':3
    }
    dataset = dmr.create_dataset(symbols[0:params['stocks']+1], params['lookback'])
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

    # todaydata = dmr.get_todaydata(lookback=5, refresh=False)
    # mdm.predict_today(model, todaydata, batch_size)
    return


def M4T22C2():
    params = {
        'model_name':'M4T22C2',
        'lookback':22,
        'batch_size':128,
        'epoch':60,
        'stocks':50,
        'indim':0,
        'outdim':2,
        'cat_func':dmr.catnorm_data2test
    }
    dataset = dmr.create_dataset(symbols[0:params['stocks']+1], params['lookback'])
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

    # todaydata = dmr.get_todaydata(lookback=5, refresh=False)
    # mdm.predict_today(model, todaydata, batch_size)
    return

def M1T5C3():
    params = {
        'model_name': 'M1T5C3_2',
        'lookback': 5,
        'batch_size': 256,
        'epoch': 120,
        'stocks': 100,
        'indim': 0,
        'outdim': 3,
        'cat_func': dmr.catnorm_data
    }
    dataset = dmr.create_dataset(symbols[0:params['stocks'] + 1], params['lookback'])
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
        'lookback': 5,
        'batch_size': 256,
        'epoch': 120,
        'stocks': 100,
        'indim': 0,
        'outdim': 3,
        'cat_func': dmr.catnorm_data
    }
    dataset = dmr.create_dataset(symbols[0:params['stocks'] + 1], params['lookback'])
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
        'lookback': 5,
        'batch_size': 256,
        'epoch': 100,
        'stocks': 300,
        'indim': 0,
        'outdim': 3,
        'cat_func': dmr.catnorm_data
    }
    # dmr.get_bsdata(True)
    dataset = dmr.create_dataset(symbols[0:params['stocks'] + 1], params['lookback'])
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


def M1T5C3_D1():
    params = {
        'model_name': 'M1T5C3_D1',
        'lookback': 5,
        'batch_size': 256,
        'epoch': 100,
        'stocks': 3100,
        'indim': 0,
        'outdim': 3,
        'cat_func': dmr.catnorm_data
    }
    # dmr.get_bsdata(True)
    dataset = dmr.create_dataset(params['stocks'], params['lookback'])
    train, test = dmr.split_dataset(dataset, 0.75, params['batch_size'])
    bsdata_train, tsdata_train, rtdata_train, lbdata_train, tsdata_train_v, rtdata_train_v, lbdata_train_v, tsdata_train_f= dmr.create_feeddata(train)
    bsdata_test, tsdata_test, rtdata_test, lbdata_test, tsdata_test_v, rtdata_test_v, lbdata_test_v, tsdata_test_f  = dmr.create_feeddata(test)
    train_x = tsdata_train_f[:,1:,1:]
    test_x = tsdata_test_f[:,1:,1:]
    train_y = params['cat_func'](lbdata_train_v[:, -2])
    test_y = params['cat_func'](lbdata_test_v[:,-2])
    params['indim'] = train_x.shape[train_x.ndim - 1]

    path = 'models/' + params['model_name']
    model_dir = os.path.dirname(path+'/logs/')
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    callbacks = [
        EarlyStopping(monitor='top_t1p1_class', patience=6, verbose=0),
        ModelCheckpoint(path+'logs', monitor='val_loss', save_best_only=True, verbose=0),
        TensorBoard(log_dir=path+'/logs', histogram_freq=0, write_graph=True, write_images=False),
    ]
    model = mdm.build_model(params)
    mdm.train_model(model, params, train_x, train_y, callbacks,  test_x, test_y)
    # model = load_model('./models/model.h5',custom_objects={'top_t1p1_class':mdm.top_t1p1_class})
    mdm.predict(model, test_x, np.hstack([rtdata_test_v,lbdata_test_v[:,2:]]), params['batch_size'], params['model_name'])

def M1T5C3_002():
    params = {
        'model_name': 'M1T5C3_002',
        'lookback': 5,
        'batch_size': 256,
        'epoch': 100,
        'stocks': 3100,
        'indim': 0,
        'outdim': 3,
        'cat_func': dmr.catnorm_data
    }
    # dmr.get_bsdata(True)
    dataset = dmr.create_dataset(params['stocks'], params['lookback'])
    train, test = dmr.split_dataset(dataset, 0.75, params['batch_size'])
    bsdata_train, tsdata_train, rtdata_train, lbdata_train, tsdata_train_v, rtdata_train_v, lbdata_train_v, tsdata_train_f= dmr.create_feeddata_hp(train)
    bsdata_test, tsdata_test, rtdata_test, lbdata_test, tsdata_test_v, rtdata_test_v, lbdata_test_v, tsdata_test_f  = dmr.create_feeddata_hp(test)
    train_x = tsdata_train_f[:,1:,1:]
    test_x = tsdata_test_f[:,1:,1:]

    print 'lbdata_train_v: ' + str(lbdata_train[lbdata_train_v[:, -2]>11])
    print 'lbdata_test_v： ' + str(lbdata_test[lbdata_test_v[:, -2]>11])
    # return
    train_y = params['cat_func'](lbdata_train_v[:, -2])
    test_y = params['cat_func'](lbdata_test_v[:,-2])
    params['indim'] = train_x.shape[train_x.ndim - 1]

    path = 'models/' + params['model_name']
    model_dir = os.path.dirname(path+'/logs/')
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    callbacks = [
        EarlyStopping(monitor='top_t1p1_class', patience=5, verbose=0),
        ModelCheckpoint(path+'/.model.h5', monitor='val_loss', save_best_only=True, verbose=0),
        TensorBoard(log_dir=path+'/tensorboard_logs', histogram_freq=0, write_graph=True, write_images=False),
    ]
    model = mdm.build_model(params)
    mdm.train_model(model, params, train_x, train_y, callbacks,  test_x, test_y)
    # model = load_model('./models/model.h5',custom_objects={'top_t1p1_class':mdm.top_t1p1_class})
    mdm.predict(model, test_x, np.hstack([rtdata_test_v,lbdata_test_v[:,2:]]), params['batch_size'], params['model_name'])


def M1T5C4():
    params = {
        'model_name': 'M1T5C4',
        'lookback': 5,
        'batch_size': 256,
        'epoch': 100,
        'stocks': 3100,
        'indim': 0,
        'outdim': 4,
        'cat_func': dmr.catnorm_data4
    }
    # dmr.get_bsdata(True)
    dataset = dmr.create_dataset(params['stocks'], params['lookback'])
    train, test = dmr.split_dataset(dataset, 0.75, params['batch_size'])
    bsdata_train, tsdata_train, rtdata_train, lbdata_train, tsdata_train_v, rtdata_train_v, lbdata_train_v, tsdata_train_f= dmr.create_feeddata_hp(train)
    bsdata_test, tsdata_test, rtdata_test, lbdata_test, tsdata_test_v, rtdata_test_v, lbdata_test_v, tsdata_test_f  = dmr.create_feeddata_hp(test)
    train_x = tsdata_train_f[:,1:,1:]
    test_x = tsdata_test_f[:,1:,1:]

    print 'lbdata_train_v: ' + str(lbdata_train[lbdata_train_v[:, -2]>11])
    print 'lbdata_test_v： ' + str(lbdata_test[lbdata_test_v[:, -2]>11])
    # return
    train_y = params['cat_func'](lbdata_train_v[:, -2])
    test_y = params['cat_func'](lbdata_test_v[:,-2])
    params['indim'] = train_x.shape[train_x.ndim - 1]

    path = 'models/' + params['model_name']
    model_dir = os.path.dirname(path+'/logs/')
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    callbacks = [
        EarlyStopping(monitor='top_t2p1_class', patience=10, verbose=0),
        ModelCheckpoint(path+'/.model.h5', monitor='val_loss', save_best_only=True, verbose=0),
        TensorBoard(log_dir=path+'/tensorboard_logs', histogram_freq=0, write_graph=True, write_images=False),
    ]
    model = mdm.build_model_T2P1(params)
    mdm.train_model(model, params, train_x, train_y, callbacks,  test_x, test_y)
    # model = load_model('./models/model.h5',custom_objects={'top_t1p1_class':mdm.top_t1p1_class})
    mdm.predict(model, test_x, np.hstack([rtdata_test_v,lbdata_test_v[:,2:]]), params['batch_size'], params['model_name'])


def predict_today():
    # M1T5B256C3
    # build_model1, lookback=5, batchsize = 256, cat_function = dmr.cat3
    params = {
        'model_name': 'M1T5C2_D2',
        'lookback': 5,
        'batch_size': 256,
        'epoch': 100,
        'stocks': 1000,
        'indim': 0,
        'outdim': 3,
        'cat_func': dmr.catnorm_data
    }
    lookback = 5
    # get today's data
    bsdata, tsdata, rtdata, tsdata_v, rtdata_v, tsdata_f = dmr.create_today_dataset(lookback)
    model = load_model('./models/model.h5',custom_objects={'top_t1p1_class':mdm.top_t1p1_class})
    if 20 > datetime.datetime.now().hour > 9:
        out = mdm.predict(model, tsdata_f[:,-lookback:,1:], rtdata_v, params['batch_size'], params['model_name'])
    else:
        out = mdm.predict(model, tsdata_f[:, -lookback-1:-1, 1:], rtdata_v, params['batch_size'], params['model_name'])

    out = out[out[:,2]>0.7][0:20][:,(2,3,-4)]
    candidates = {}
    for s in out:
        candidates[int2str(s[1])] = s[-1]
    return candidates


def predict_today_M1T5C4():
    # M1T5B256C3
    # build_model1, lookback=5, batchsize = 256, cat_function = dmr.cat3
    params = {
        'model_name': 'M1T5C4',
        'lookback': 5,
        'batch_size': 256,
        'epoch': 100,
        'stocks': 3100,
        'indim': 0,
        'outdim': 4,
        'cat_func': dmr.catnorm_data4
    }
    lookback = 5
    # get today's data
    bsdata, tsdata, rtdata, tsdata_v, rtdata_v, tsdata_f = dmr.create_today_dataset(lookback)
    model = load_model('./models/model.h5',custom_objects={'top_t2p1_class':mdm.top_t2p1_class})
    if 20 > datetime.datetime.now().hour > 9:
        out = mdm.predict(model, tsdata_f[:,-lookback:,1:], rtdata_v, params['batch_size'], params['model_name'])
    else:
        out = mdm.predict(model, tsdata_f[:, -lookback-1:-1, 1:], rtdata_v, params['batch_size'], params['model_name'])

    out = out[out[:,2]>0.7][0:20][:,(2,3,-4)]
    candidates = {}
    for s in out:
        candidates[int2str(s[1])] = s[-1]
    return candidates


def MTODAY():
    try:
        model = load_model('./models/model.h5',custom_objects={'top_t1p1_class':mdm.top_t1p1_class})
    except:
        raise "Can't load model at: ./models/model.h5"
    todaydata = dmr.get_todaydata(lookback=5, refresh=False)
    mdm.predict(model, todaydata, 256)


def _main_():
    for i in range(1, len(sys.argv)):
        eval(sys.argv[i])()

if __name__ == '__main__':
    _main_()
    # M1T5C3_D1()
    # predict_today()
