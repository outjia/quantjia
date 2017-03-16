# coding=utf-8

import datetime
import os
import numpy as np
import pandas as pd
from DataManager import *
from ModelManager import *
from DataManager import int2str
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from datetime import date
from datetime import timedelta
import sys


def parse_params(mstr):
    # M1_T5_B256_C3_E100_S3100_
    # build_model1, lookback=5, batch_size = 256, catf = catnorm_data, epoch=100, stocks = 3100,

    catf = {'C3':'catf3', 'C4':'catf4'}
    models = {'M1':'build_model', 'M2':'build_model2', 'M3':'build_model3'}
    custmetrics = {'C3':'top_t1p1', 'C4':'top_t2p1'}
    params = {}
    params['model_name'] = mstr
    mstr_arr = str(mstr).split('_')
    for s in mstr_arr:
        if s.startswith('M'):
            params['model'] = models[s]
        if s.startswith('T'):
            params['lookback'] = int(s[1:])
        if s.startswith('B'):
            params['batch_size'] = int(s[1:])
        if s.startswith('C'):
            params['catf'] = catf[s]
            params['outdim'] = int(s[1:])
            params['custmetric'] = custmetrics[s]
        if s.startswith('E'):
            params['epoch'] = int(s[1:])
        if s.startswith('S'):
            params['stocks'] = int(s[1:])
    return params


def train_model(mstr, days=None):
    params = parse_params(mstr)
    print ("[ train model ]... " + mstr)
    dataset = create_dataset(params['stocks'], params['lookback'], days, 8)
    train, test = split_dataset(dataset, 0.75, params['batch_size'])
    bstrain, tstrain, lbtrain_v = create_feeddata(train)
    bstest, tstest, lbtest_v = create_feeddata(test)
    train_x = tstrain[:,:,1:]
    train_y = eval(params['catf'])(lbtrain_v[:, 2])
    train_y, train_x, non = balance_data(train_y, train_x)
    sz = len(train_y)/params['batch_size'] * params['batch_size']
    train_x = train_x[:sz]
    train_y = train_y[:sz]

    test_x = tstest[:,:,1:]
    test_y = eval(params['catf'])(lbtest_v[:,2])
    test_y_v = lbtest_v

    params['indim'] = train_x.shape[train_x.ndim - 1]
    path = 'models/' + params['model_name']
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min'),
        ModelCheckpoint(path+'/checkpoint_model.h5', monitor='val_'+params['custmetric'], save_best_only=True, verbose=0),
        TensorBoard(log_dir=path+'/tensorboard_logs', histogram_freq=0, write_graph=True, write_images=False),
    ]
    model = eval(params['model'])(params)
    model.fit(train_x, train_y, batch_size=params['batch_size'], nb_epoch=params['epoch'],
              validation_data=(test_x, test_y), callbacks=callbacks)
    save_model(model, './models/' + params['model_name'] + '/model.h5')

    proba = model.predict_proba(test_x, verbose=0, batch_size=params['batch_size'])
    out = np.hstack([proba, test_y_v])
    sortout = out[(-out[:, proba.shape[proba.ndim - 1] - 1]).argsort(), :]

    if not __debug__:
        np.savetxt("./models/" + params['model_name'] + "/val_result.txt", sortout, fmt='%f')
    else:
        print sortout[0:200, :]
    print "[ End train model ]"
    return sortout


def train_model2(mstr, days=None):
    params = parse_params(mstr)
    print ("[ train model ]... " + mstr)
    dataset = create_dataset(params['stocks'], params['lookback'], days)
    train, test = split_dataset(dataset, 0.75, params['batch_size'])
    bstrain, tstrain, lbtrain_v = create_feeddata(train)
    bstest, tstest, lbtest_v = create_feeddata(test)
    train_x1 = tstrain[:,:,1:]
    train_x2 = bstrain[:,1:]
    train_y = eval(params['catf'])(lbtrain_v[:, 2])
    train_y, train_x1, train_x2 = balance_data(train_y, train_x1, train_x2)
    sz = len(train_y)/params['batch_size'] * params['batch_size']
    train_x1 = train_x1[:sz]
    train_x2 = train_x2[:sz]
    train_y = train_y[:sz]

    test_x1 = tstest[:,:,1:]
    test_x2 = bstest[:, 1:]
    test_y = eval(params['catf'])(lbtest_v[:,2])
    test_y_v = lbtest_v

    params['indim1'] = train_x1.shape[-1]
    params['indim2'] = train_x2.shape[-1]
    path = 'models/' + params['model_name']
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min'),
        ModelCheckpoint(path+'/checkpoint_model.h5', monitor='val_'+params['custmetric'], save_best_only=True, verbose=0),
        TensorBoard(log_dir=path+'/tensorboard_logs', histogram_freq=0, write_graph=True, write_images=False),
    ]
    model = eval(params['model'])(params)
    model.fit([train_x1, train_x2], train_y, batch_size=params['batch_size'], nb_epoch=params['epoch'],
              validation_data=([test_x1,test_x2], test_y), callbacks=callbacks)
    save_model(model, './models/' + params['model_name'] + '/model.h5')

    proba = model.predict_proba([test_x1,test_x2], verbose=0, batch_size=params['batch_size'])
    out = np.hstack([proba, test_y_v])
    sortout = out[(-out[:, proba.shape[proba.ndim - 1] - 1]).argsort(), :]

    if not __debug__:
        np.savetxt("./models/" + params['model_name'] + "/val_result.txt", sortout, fmt='%f')
    else:
        print sortout[0:200, :]
    print "[ End train model ]"
    return sortout


def validate_model(mstr, days=8):
    print ("[ valid model: %s ]... with latest %s days data"%(mstr,str(days)))
    params = parse_params(mstr)
    cust_objs = {params['custmetric']:eval(params['custmetric']), 'top_t1p1':top_t1p1}
    model = load_model('./models/'+params['model_name']+'/model.h5',custom_objects=cust_objs)

    valset = create_dataset(3100, params['lookback'], days, 8)
    bsvalset, tsvalset, lbvalset_v = create_feeddata(valset)

    test_x = tsvalset[:,:,1:]
    test_y_v = lbvalset_v

    if model.stateful:
        test_x = test_x[:len(test_x) / params['batch_size'] * params['batch_size']]
        test_y_v = test_y_v[:len(test_y_v) / params['batch_size'] * params['batch_size']]
    proba = model.predict_proba(test_x, verbose=0, batch_size=params['batch_size'])
    out = np.hstack([proba, test_y_v])
    sortout = out[(-out[:, proba.shape[proba.ndim - 1] - 1]).argsort(), :]
    if not __debug__:
        np.savetxt("./models/" + params['model_name'] + "/val_newlydata_result_"+str(days)+".txt", sortout, fmt='%f')
    else:
        print sortout[0:200, :]

    # if os.path.exists('./models/' + params['model_name'] + '/checkpoint_model.h5'):
    #     model_bk = load_model('./models/' + params['model_name'] + '/checkpoint_model.h5', custom_objects=cust_objs)
    #     proba_bk = model_bk.predict_proba(test_x, verbose=0, batch_size=params['batch_size'])
    #     out_bk = np.hstack([proba_bk, test_y_v])
    #     sortout_bk = out_bk[(-out_bk[:, proba_bk.shape[proba_bk.ndim - 1] - 1]).argsort(), :]
    #     np.savetxt("./models/" + params['model_name'] + "/val_cpt_newlydata_result_"+str(days)+".txt", sortout_bk, fmt='%f')
    bins = np.arange(0,1,0.1)
    idx = params['outdim'] - 1
    labels = bins.searchsorted(sortout[:, idx])
    print pd.Series(sortout[:, -2]).groupby(labels).mean()

    print ("[ End validate model: %s ]... " % (mstr))
    return sortout


def validate_model2(mstr, days=8):
    print ("[ valid model: %s ]... with latest %s days data"%(mstr,str(days)))
    params = parse_params(mstr)
    cust_objs = {params['custmetric']:eval(params['custmetric']), 'top_t1p1':top_t1p1}
    model = load_model('./models/'+params['model_name']+'/model.h5',custom_objects=cust_objs)

    valset = create_dataset(3100, params['lookback'], days=days)
    bsvalset, tsvalset, lbvalset_v = create_feeddata(valset)

    test_x = tsvalset[:,:,1:]
    test_x2 = bsvalset[:,1:]
    test_y_v = lbvalset_v
    test_x = test_x[:len(test_x) / params['batch_size'] * params['batch_size']]
    test_x2 = test_x2[:len(test_x2) / params['batch_size'] * params['batch_size']]
    test_y_v = test_y_v[:len(test_y_v) / params['batch_size'] * params['batch_size']]

    proba = model.predict_proba([test_x,test_x2], verbose=0, batch_size=params['batch_size'])
    out = np.hstack([proba, test_y_v])
    sortout = out[(-out[:, proba.shape[proba.ndim - 1] - 1]).argsort(), :]
    if not __debug__:
        np.savetxt("./models/" + params['model_name'] + "/val_newlydata_result.txt", sortout, fmt='%f')
    else:
        print sortout[0:200, :]

    if os.path.exists('./models/' + params['model_name'] + '/checkpoint_model.h5'):
        model_bk = load_model('./models/' + params['model_name'] + '/checkpoint_model.h5', custom_objects=cust_objs)
        proba_bk = model_bk.predict_proba([test_x,test_x2], verbose=0, batch_size=params['batch_size'])
        out_bk = np.hstack([proba_bk, test_y_v])
        sortout_bk = out_bk[(-out_bk[:, proba_bk.shape[proba_bk.ndim - 1] - 1]).argsort(), :]
        np.savetxt("./models/" + params['model_name'] + "/val_checkpoint_newlydata_result.txt", sortout_bk, fmt='%f')

    print ("[ End validate model: %s ]... " % (mstr))
    return sortout


def predict_batch():
    file = os.path.realpath(__file__)
    os.chdir(os.path.dirname(file))
    print os.path.dirname(file)
    path = "./models/confirm/"
    for model in os.listdir(path):
        if os.path.isdir(path+model):
            if os.path.exists(path+model + '/model.h5'):
                predict_today(model, path)


def predict_today(mstr, path='./models/'):
    print ("[ select stocks ]... using model:" + mstr)
    params = parse_params(mstr)
    cust_objs = {params['custmetric']:eval(''+params['custmetric']), 'top_t1p1':top_t1p1}
    model = load_model(path+params['model_name']+'/model.h5',custom_objects=cust_objs)

    tsdata, rtdata_v = create_today_dataset(params['lookback'])
    batch_size = params['batch_size']
    data_x = tsdata[:len(tsdata) / batch_size * batch_size]
    data_v = rtdata_v[:len(rtdata_v) / batch_size * batch_size]
    proba = model.predict_proba(data_x, verbose=0, batch_size=batch_size)
    out = np.column_stack([proba, data_v[:,:]])
    sortout = out[(-out[:, proba.shape[proba.ndim - 1] - 1]).argsort(), :]

    if not __debug__:
        dt = (datetime.date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
        np.savetxt(path + params['model_name'] + "/"+dt+"_result.txt", sortout, fmt='%f')
    else:
        print sortout[0:20, :]

    idx = params['outdim'] - 1
    sortout = sortout[sortout[:,idx]>=0.5][:,(idx+1, -4, idx)]
    candidates = pd.DataFrame(sortout, columns=('code','price', 'proba'))
    print "[ End prediction ] of tomorrow's price"
    return candidates


def _main_():
    if(len(sys.argv) > 3):
        eval(sys.argv[1])(sys.argv[2], int(sys.argv[3]))
    elif(len(sys.argv) > 2):
        eval(sys.argv[1])(sys.argv[2])
    else:
        eval(sys.argv[1])()

if __name__ == '__main__':
    _main_()
    # M1T5C3_D1()
    # predict_today()
