# coding=utf-8

import datetime
import os
import numpy as np
import pandas as pd
import DataManager as dm
import ModelManager as mdm
from DataManager import int2str
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import sys

dmr = dm.DataManager()


def parse_params(mstr):
    # M1_T5_B256_C3_E100_S3100_
    # build_model1, lookback=5, batch_size = 256, catf = dmr.catnorm_data, epoch=100, stocks = 3100,

    catf = {'C3':'dmr.catnorm_data', 'C4':'dmr.catnorm_data4'}
    models = {'M1':'mdm.build_model', 'M2':'mdm.build_model2'}
    custmetrics = {'C3':'top_t1p1_class', 'C4':'top_t2p1_class'}
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


def train_model_simple(mstr):
    params = parse_params(mstr)
    dataset = dmr.create_dataset_simple(params['stocks'], params['lookback'])
    train, test = dmr.split_dataset(dataset, 0.75, params['batch_size'])
    bstrain, tstrain, lbtrain_v = dmr.create_feeddata_hp_simple(train)
    bstest, tstest, lbtest_v = dmr.create_feeddata_hp_simple(test)

    train_x = tstrain[:,:,1:]
    test_x = tstest[:,:,1:]
    train_y = eval(params['catf'])(lbtrain_v[:, -2])
    test_y = eval(params['catf'])(lbtest_v[:,-2])
    test_y_v = lbtest_v

    params['indim'] = train_x.shape[train_x.ndim - 1]
    path = 'models/' + params['model_name']
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, verbose=0, mode='min'),
        ModelCheckpoint(path+'/model.h5', monitor='val_loss', save_best_only=True, verbose=0),
        TensorBoard(log_dir=path+'/tensorboard_logs', histogram_freq=0, write_graph=True, write_images=False),
    ]
    model = eval(params['model'])(params)
    model.fit(train_x, train_y, batch_size=params['batch_size'], nb_epoch=params['epoch'],
              validation_data=(test_x, test_y), callbacks=callbacks)
    if model.stateful:
        test_x = test_x[:len(test_x) / params['batch_size'] * params['batch_size']]
        test_y_v = test_y_v[:len(test_y_v) / params['batch_size'] * params['batch_size']]
    proba = model.predict_proba(test_x, verbose=0, batch_size=params['batch_size'])

    out = np.hstack([proba, test_y_v])
    sortout = out[(-out[:, proba.shape[proba.ndim - 1] - 1]).argsort(), :]

    if not __debug__:
        np.savetxt("./models/" + params['model_name'] + "/val_result.txt", sortout, fmt='%f')
    else:
        print sortout[0:200, :]
    return sortout


def predict_today_simple(mstr):
    params = parse_params(mstr)
    cust_objs = {params['custmetric']:eval('mdm.'+params['custmetric'])}
    tsdata, rtdata_v = dmr.create_today_dataset_simple(params['lookback'])
    model = load_model('./models/'+params['model_name']+'/model.h5',custom_objects=cust_objs)

    # if model.stateful:
    batch_size = params['batch_size']
    # else:
    #     batch_size = 32
    data_x = tsdata[:len(tsdata) / batch_size * batch_size]
    data_v = rtdata_v[:len(rtdata_v) / batch_size * batch_size]
    proba = model.predict_proba(data_x, verbose=0, batch_size=batch_size)
    out = np.column_stack([proba, data_v[:,:]])
    sortout = out[(-out[:, proba.shape[proba.ndim - 1] - 1]).argsort(), :]

    if not __debug__:
        np.savetxt("./models/" + params['model_name'] + "/today_result_simple.txt", sortout, fmt='%f')
    else:
        print sortout[0:20, :]

    idx = params['outdim'] - 1
    sortout = sortout[sortout[:,idx]>=0.5][:,(idx+1, -4, idx)]
    candidates = pd.DataFrame(sortout, columns=('code','price', 'proba'))
    return candidates


def validate_model(mstr, days=8):
    params = parse_params(mstr)
    cust_objs = {params['custmetric']:eval('mdm.'+params['custmetric'])}
    bsdata, tsdata, tsdata_v, lbdata_v = dmr.create_val_dataset(params['lookback'], days=days)
    model = load_model('./models/'+params['model_name']+'/model.h5',custom_objects=cust_objs)

    batch_size = params['batch_size']
    day = 1
    while day <= days:
        data_x = tsdata[:,-params['lookback']-day:-day,1:]
        data_x = data_x[:len(data_x) / batch_size * batch_size]
        data_v = lbdata_v[:len(lbdata_v) / batch_size * batch_size]
        tsdata_v = tsdata_v[:len(tsdata_v) / batch_size * batch_size]
        proba = model.predict_proba(data_x, verbose=0, batch_size=batch_size)
        out = np.column_stack([proba, tsdata_v[:,-day-1,(0,-2)], data_v[:,-day,:]])
        out = out[out[:,proba.shape[-1] + 1]<9.98]
        sortout = out[(-out[:, proba.shape[proba.ndim - 1] - 1]).argsort(), :]
        if not __debug__:
            np.savetxt("./models/" + params['model_name'] + "/val_d" +str(day)+ "_result.txt", sortout, fmt='%f')
        else:
            print sortout[0:20, :]
        day += 1


def _main_():
    if(len(sys.argv) > 2):
        eval(sys.argv[1])(sys.argv[2])
    else:
        eval(sys.argv[1])()

if __name__ == '__main__':
    _main_()
    # M1T5C3_D1()
    # predict_today()
