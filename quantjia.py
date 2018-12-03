# coding=utf-8

from __future__ import absolute_import

import json
import sys
from SimpleXMLRPCServer import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
from SocketServer import TCPServer

from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.utils import plot_model

from DM import *
from ModelManager import *


def parse_params(mstr):
    # M1_T5_B256_C3_E100_Lmin_Mgem_K5
    # build_model1, lookback=5, batch_size = 256, catf = catnorm_data, epoch=100, label = 'min', mem = 'gem'

    catf = {'C3': 'catf3', 'C4': 'catf4', 'C2': 'catf2', 'C22': 'catf22', 'C31': 'catf31', 'C32': 'catf32', 'C1': 'noncatf','C20': 'catf20'}
    models = {'M1': 'build_model', 'M2': 'build_model2', 'M3': 'build_model3', 'M4': 'build_model4', 'M5': 'build_model5', 'MR': 'nbuild_rmodel', 'MLR': 'nbuild_lrmodel'}
    params = {}
    params['mclass'] = ''
    params['model_name'] = mstr
    params['metrics'] = ['categorical_accuracy']
    params['cmetrics'] = {'recall': recall, 'top1_recall': top1_recall, 'top_t1p1': top_t1p1}
    params['main_metric'] = {'top_t1p1': top_t1p1}
    params['totals'] = 5

    mstr_arr = str(mstr).upper().split('_')
    for s in mstr_arr:
        if s.startswith('M'):
            params['model'] = models[s]
            params['mclass'] = params['mclass'] + s
        if s.startswith('T'):
            params['lookback'] = int(s[1:])
        if s.startswith('B'):
            params['batch_size'] = int(s[1:])
        if s.startswith('C'):
            params['catf'] = catf[s]
            params['mclass'] = params['mclass'] + s[0:2]
            params['outdim'] = int(s[1:2])
            if int(s[1:2]) >= 3:
                params['cmetrics']['top_t2p1'] = top_t2p1
                # params['cmetrics']['top2_recall'] = top2_recall
                # params['main_metric'] = {'top_t2p1': top_t2p1}
            if int(s[1:2]) == 1:
                params['main_metric'] = {'mse'}

        if s.startswith('E'):
            params['epoch'] = int(s[1:])
        if s.startswith('L'):
            params['mclass'] = params['mclass'] + s
            params['label'] = s[1:].lower()
        if s.startswith('M'):
            params['mem'] = s[1:]
        if s.startswith('K'):
            params['ktype'] = s[1:]
    params['metrics'].extend(params['cmetrics'].values())
    return params


def ntrain_model(mstr, start, mid, end):
    params = parse_params(mstr)
    print ("[ train model ]... " + mstr)

    labcol_map = {'o2c': -4, 'close': -3, 'min': -2, 'max': -1}
    labelcol = labcol_map[params['label']]
    index = ['basic']
    # index = ['sme', 'gem']
    # index = None
    if __debug__:
        index = ['basic']

    dataset = ncreate_dataset(days=params['lookback'], index=index, start=start, end=mid, ktype=params['ktype'])
    test = ncreate_dataset(index=index, days=params['lookback'], start=mid, end=end, ktype=params['ktype'])

    train, val = split_dataset(dataset, 0.75, params['batch_size'], 1535712851.0)
    bstrain, tstrain, lbtrain_v = create_feeddata(train)
    bsval, tsval, lbval_v = create_feeddata(val)
    bstest, tstest, lbtest_v = create_feeddata(test)

    train_x = tstrain
    train_y = eval(params['catf'])(lbtrain_v[:, labelcol])
    train_y, train_x, tmp = balance_data(train_y, train_x)
    sz = len(train_y) // params['batch_size'] * params['batch_size']
    train_x = train_x[:sz]
    train_y = train_y[:sz]

    val_x = tsval
    val_y = eval(params['catf'])(lbval_v[:, labelcol])

    sz = len(lbtest_v) // params['batch_size'] * params['batch_size']
    test_x = tstest[:sz]
    test_y = eval(params['catf'])(lbtest_v[:sz, labelcol])
    test_y_v = lbtest_v[:sz]

    params['indim'] = train_x.shape[train_x.ndim - 1]

    path = 'models/' + params['mclass'] + '/' + params['model_name']
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir): os.makedirs(model_dir)

    if __debug__:
        patience = 10
    else:
        patience = 200

    logdir = datetime.datetime.now().strftime(path + '/S' + start + '.%Y%m%d.%H.%M.%S.run')
    if not os.path.exists(logdir): os.makedirs(logdir)
    path = logdir

    callbacks = [
        EarlyStopping(monitor='val_top_t1p1', patience=patience, verbose=0, mode='max'),
        ModelCheckpoint(path + '/weights.{epoch:02d}-{val_top_t1p1:.2f}.hdf5', monitor='val_' + params['main_metric'].keys()[0], period=10, verbose=0, mode='max'),
        TensorBoard(log_dir=path + '/tensorboard_logs', histogram_freq=0, write_graph=True, write_images=False),
    ]
    model = eval(params['model'])(params)
    print ("model summary")
    model.summary()
    model.fit(train_x, train_y, batch_size=params['batch_size'], epochs=params['epoch'], validation_data=(val_x, val_y), callbacks=callbacks)
    save_model(model, path + '/model.h5')

    if params['catf'] == 'noncatf':
        return None
    else:
        print ('使用最优模型进行预测')
        model = load_model(path + '/best_model.h5', custom_objects=params['cmetrics'])
        proba = model.predict_proba(test_x, verbose=0, batch_size=params['batch_size'])

        out = np.hstack([proba, test_y_v])
        sortout = out[(-out[:, proba.shape[-1] - 1]).argsort(), :]
        if not __debug__:
            np.savetxt(path + "/val_result.txt", sortout, fmt='%f')

        print_dist_cut(sortout, proba.shape[-1] - 1, labelcol, 20, path)
        print_dist(sortout, proba.shape[-1] - 1, labelcol, 10)
    print ("[ End train model ]")
    return sortout


def ntrain_model2(mstr, start, end, train_step=30, val_step=15):
    params = parse_params(mstr)
    print ("[ train model ]... " + mstr)

    labcol_map = {'o2c': -4, 'close': -3, 'min': -2, 'max': -1}
    labelcol = labcol_map[params['label']]
    index = None
    # index = ['sme', 'gem']
    if __debug__:
        index = ['debug']

    path = 'models/' + params['mclass'] + '/' + params['model_name']
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir): os.makedirs(model_dir)

    if __debug__:
        patience = 2
    else:
        patience = 200

    train_step = int(train_step)
    val_step = int(val_step)
    start_dt = datetime.datetime.strptime(start, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end, '%Y-%m-%d')
    mid_dt = start_dt + timedelta(train_step)
    end_dt1 = mid_dt + timedelta(val_step)

    result_date = []
    result_data = []
    path_m = path
    while mid_dt < end_dt:
        start_str = start_dt.strftime('%Y-%m-%d')
        mid_str = mid_dt.strftime('%Y-%m-%d')
        end_str = end_dt1.strftime('%Y-%m-%d')

        dataset = ncreate_dataset(index=index, days=params['lookback'], start=start_str, end=mid_str, ktype=params['ktype'])
        test = ncreate_dataset(index=index, days=params['lookback'], start=mid_str, end=end_str, ktype=params['ktype'])

        # dataset = ncreate_dataset(index=None,days=params['lookback'], start=start, end=end, ktype=params['ktype'])
        # train, test = split_dataset(dataset, 0.75, params['batch_size'])

        train, val = split_dataset(dataset, 0.75, params['batch_size'])
        bstrain, tstrain, lbtrain_v = create_feeddata(train)
        bsval, tsval, lbval_v = create_feeddata(val)
        bstest, tstest, lbtest_v = create_feeddata(test)

        train_x = tstrain
        train_y = eval(params['catf'])(lbtrain_v[:, labelcol])
        train_y, train_x, tmp = balance_data(train_y, train_x)
        sz = len(train_y) // params['batch_size'] * params['batch_size']
        train_x = train_x[:sz]
        train_y = train_y[:sz]
        print (np.sum(train_y, axis=0))


        val_x = tsval
        val_y = eval(params['catf'])(lbval_v[:, labelcol])
        print (np.sum(val_y, axis=0))

        sz = len(lbtest_v) // params['batch_size'] * params['batch_size']
        test_x = tstest[:sz]
        test_y = eval(params['catf'])(lbtest_v[:sz, labelcol])
        test_y_v = lbtest_v[:sz]

        params['indim'] = train_x.shape[train_x.ndim - 1]

        logdir = datetime.datetime.now().strftime(path_m + '/S' + start_str + 'T%Y%m%d.%H.%M.%S.run')
        if not os.path.exists(logdir): os.makedirs(logdir)
        path = logdir

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, verbose=0, mode='min'),
            ModelCheckpoint(path + '/best_model.h5', monitor='val_' + params['main_metric'].keys()[0], save_best_only=True, verbose=0, mode='max'),
            TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=False, write_images=False),
        ]
        model = eval(params['model'])(params)
        print ("model summary")
        model.summary()
        print (np.sum(train_y, axis=0))
        print (np.sum(val_y, axis=0))
        model.fit(train_x, train_y, batch_size=params['batch_size'], epochs=params['epoch'],  # validation_split=0.33,callbacks=callbacks)
                  validation_data=(val_x, val_y), callbacks=callbacks)
        save_model(model, path + '/model.h5')

        if params['catf'] == 'noncatf':
            return None
        else:
            print ('使用最优模型进行预测: S=' + start_str)
            model = load_model(path + '/best_model.h5', custom_objects=params['cmetrics'])
            proba = model.predict_proba(test_x, verbose=0, batch_size=params['batch_size'])

            out = np.hstack([proba, test_y_v])
            sortout = out[(-out[:, proba.shape[-1] - 1]).argsort(), :]
            if not __debug__:
                np.savetxt(path + "/val_result.txt", sortout, fmt='%f')

            print_dist_cut(sortout, proba.shape[-1] - 1, labelcol, 20, path)
            print_dist(sortout, proba.shape[-1] - 1, labelcol, 10)
            result_date.append(start_str)
            result_data.append(sortout)

        start_dt = mid_dt
        mid_dt = start_dt + timedelta(train_step)
        end_dt1 = mid_dt + timedelta(val_step)
        if mid_dt + timedelta(train_step + val_step) >= end_dt:
            end_dt1 = end_dt
    i = 0
    while i < len(result_date):
        print ('================' + result_date[i] + '==================')
        # print_dist(result_data[i], params['outdim']-1, labelcol, 2)
        print_dist(result_data[i], params['outdim'] - 1, labelcol, 10)
        # print_dist_cut(result_data[i], params['outdim'], labelcol, 20)
        i += 1

    i = 0
    while i < len(result_date):
        print ('================' + result_date[i] + '==================')
        print_dist(result_data[i], params['outdim'] - 1, labelcol, 2)
        # print_dist(result_data[i], params['outdim']-1, labelcol, 10)
        # print_dist_cut(result_data[i], params['outdim'], labelcol, 20)
        i += 1

    print ("[ End train model ]")


def nvalid_model(mstr, run=None, start=(datetime.date.today() - timedelta(days=60)).strftime('%Y-%m-%d'), end=None, mname='best_model.h5'):
    print ("[ valid model: %s ]... with data from %s to %s" % (mstr, start, end))
    params = parse_params(mstr)
    path = 'models/' + params['mclass'] + '/' + params['model_name']
    if run is not None:
        path = path + '/' + run + '/'
    model = load_model(path + mname, custom_objects=params['cmetrics'])
    print ("model summary")
    model.summary()

    labcol_map = {'close': -3, 'min': -2, 'max': -1}
    labelcol = labcol_map[params['label']]

    dataset = ncreate_dataset(days=params['lookback'], start=start, end=end, ktype=params['ktype'])
    bs, ts, lb_v = create_feeddata(dataset)
    y = eval(params['catf'])(lb_v[:, labelcol])
    sz = len(y) // params['batch_size'] * params['batch_size']
    x = ts[:sz]
    y = y[:sz]
    lb_v = lb_v[:sz]

    proba = model.predict_proba(x, verbose=0, batch_size=params['batch_size'])
    out = np.hstack([proba, lb_v])
    sortout = out[(-out[:, proba.shape[- 1] - 1]).argsort(), :]

    print_dist(sortout, proba.shape[- 1] - 1, labelcol, 20)
    print_dist_cut(sortout, params['outdim'] - 1, labelcol, 20)

    if not __debug__:
        np.savetxt(path + "/best_model." + start + "." + end + ".txt", sortout, fmt='%f')
    else:
        print (sortout[0:200, :])

    print ("[ End validate model: %s ]... " % (mstr))
    # return sortout


def ntrain_lrmodel(mstr, start, mid, end):
    params = parse_params(mstr)
    print ("[ train model ]... " + mstr)

    labcol_map = {'o2c': -4, 'close': -3, 'min': -2, 'max': -1}
    labelcol = labcol_map[params['label']]
    index = ['sme','gem']
    # index = ['sme']
    # index = None
    if __debug__:
        index = ['debug']

    dataset = ncreate_dataset(days=params['lookback'], index=index, start=start, end=mid, ktype=params['ktype'])
    test = ncreate_dataset(index=index, days=params['lookback'], start=mid, end=end, ktype=params['ktype'])

    train, val = split_dataset(dataset, 0.75, params['batch_size'])
    bstrain, tstrain, lbtrain_v = create_feeddata(train)
    bsval, tsval, lbval_v = create_feeddata(val)
    bstest, tstest, lbtest_v = create_feeddata(test)

    train_x = tstrain
    train_y = eval(params['catf'])(lbtrain_v[:, labelcol])
    # train_y, train_x, tmp = balance_data(train_y, train_x)
    sz = len(train_y) // params['batch_size'] * params['batch_size']
    train_x = train_x[:sz]
    train_y = train_y[:sz]

    val_x = tsval
    val_y = eval(params['catf'])(lbval_v[:, labelcol])

    sz = len(lbtest_v) // params['batch_size'] * params['batch_size']
    test_x = tstest[:sz]
    test_y = eval(params['catf'])(lbtest_v[:sz, labelcol])
    test_y_v = lbtest_v[:sz]

    params['indim'] = train_x.shape[train_x.ndim - 1]

    path = 'models/' + params['mclass'] + '/' + params['model_name']
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir): os.makedirs(model_dir)

    if __debug__:
        patience = 2
    else:
        patience = 200

    logdir = datetime.datetime.now().strftime(path + '/S' + start + '.%Y%m%d.%H.%M.%S.run')
    if not os.path.exists(logdir): os.makedirs(logdir)
    path = logdir

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0, mode='min'),
        ModelCheckpoint(path + '/best_model.h5', monitor='val_loss', save_best_only=True, verbose=0, mode='min'),
        TensorBoard(log_dir=path, histogram_freq=0, write_graph=True, write_images=False),
    ]
    model = eval(params['model'])(params)
    print ("model summary")
    model.summary()
    model.fit(train_x, train_y, batch_size=params['batch_size'], epochs=params['epoch'], validation_data=(val_x, val_y), callbacks=callbacks)
    save_model(model, path + '/model.h5')

    print ('使用最优模型进行预测')
    model = load_model(path + '/best_model.h5')
    proba = model.predict(test_x, verbose=0, batch_size=params['batch_size'])

    out = np.hstack([proba, test_y_v])
    sortout = out[(-out[:, proba.shape[-1] - 1]).argsort(), :]
    if not __debug__:
        np.savetxt(path + "/best_model." + start + "." + end + ".txt", sortout, fmt='%f')

    print_dist_cut(sortout, proba.shape[-1] - 1, labelcol, 20, path)
    print_dist(sortout, proba.shape[-1] - 1, labelcol, 10)
    print ("[ End train model ]")


def test_model(mstr, run, start, end, test_step=30, mname='best_model.h5'):
    params = parse_params(mstr)
    print ("[ valid model ]... " + mstr)

    labcol_map = {'o2c': -4, 'close': -3, 'min': -2, 'max': -1}
    labelcol = labcol_map[params['label']]
    index = ['sme', 'gem']
    # index = ['basic']
    if __debug__:
        index = ['debug']

    params = parse_params(mstr)
    path = 'models/' + params['mclass'] + '/' + params['model_name']
    if run is not None:
        path = path + '/' + run+'/'
    model = load_model(path + mname, custom_objects=params['cmetrics'])
    print ("model summary")
    model.summary()

    test_step = int(test_step)
    start_dt = datetime.datetime.strptime(start, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end, '%Y-%m-%d')
    mid_dt = next_n_busday(start_dt, test_step)

    result_date = []
    result_data = []
    path_m = path
    while mid_dt < end_dt:
        if not isbusday(start_dt):
            start_dt = next_n_busday(start_dt, 1)
            mid_dt = next_n_busday(mid_dt, 1)
            continue
        start_str = start_dt.strftime('%Y-%m-%d')
        mid_str = mid_dt.strftime('%Y-%m-%d')

        test = ncreate_dataset(index=index, days=params['lookback'], start=start_str, end=mid_str, ktype=params['ktype'])
        if len(test) == 0:
            start_dt = next_n_busday(start_dt, 1)
            mid_dt = next_n_busday(mid_dt, 1)
            continue
        bstest, tstest, lbtest_v = create_feeddata(test)

        sz = len(lbtest_v) // params['batch_size'] * params['batch_size']
        test_x = tstest[:sz]
        test_y = eval(params['catf'])(lbtest_v[:sz, labelcol])
        test_y_v = lbtest_v[:sz]

        params['indim'] = test_x.shape[test_x.ndim - 1]
        if params['catf'] == 'noncatf':
            return None
        else:
            print ('使用最优模型进行预测: S=' + start_str)
            proba = model.predict_proba(test_x, verbose=0, batch_size=params['batch_size'])
            out = np.hstack([proba, test_y_v])
            sortout = out[(-out[:, proba.shape[-1] - 1]).argsort(), :]
            if not __debug__:
                np.savetxt(path + "/best_model." + start_str + "." + mid_str + ".txt", sortout, fmt='%f')

            print_dist_cut(sortout, proba.shape[-1] - 1, labelcol, 20, path)
            print_dist(sortout, proba.shape[-1] - 1, labelcol, 10)
            result_date.append(start_str)
            result_data.append(sortout)

        start_dt = mid_dt
        mid_dt = next_n_busday(start_dt, test_step)
    i = 0
    while i < len(result_date):
        print ('================' + result_date[i] + '==================')
        # print_dist(result_data[i], params['outdim']-1, labelcol, 2)
        print_dist(result_data[i], params['outdim'] - 1, labelcol, 10)
        # print_dist_cut(result_data[i], params['outdim'], labelcol, 20)
        i += 1

    i = 0
    while i < len(result_date):
        print ('================' + result_date[i] + '==================')
        print_dist(result_data[i], params['outdim'] - 1, labelcol, 2)
        # print_dist(result_data[i], params['outdim']-1, labelcol, 10)
        # print_dist_cut(result_data[i], params['outdim'], labelcol, 20)
        i += 1

    print ("[ End validate model ]")


def predict_batch():
    file = os.path.realpath(__file__)
    os.chdir(os.path.dirname(file))
    print (os.path.dirname(file))
    path = "./models/confirm/"
    for model in os.listdir(path):
        if os.path.isdir(path + model):
            if os.path.exists(path + model + '/model.h5'):
                predict_today(model, path)


def predict_today(mstr, run,force_return=False):
    print ("[ select stocks ]... using model:" + mstr)
    params = parse_params(mstr)
    path = 'models/' + params['mclass'] + '/' + params['model_name']
    if run is not None:
        path = path + '/' + run
    model = load_model(path + '/best_model.h5', custom_objects=params['cmetrics'])
    print ("model summary")
    model.summary()
    index = None
    index = ['sme', 'gem']
    if __debug__:
        index = ['sme']
        pass

    dataset = ncreate_today_dataset(index=index, days=params['lookback'],force_return=force_return)
    bs, ts, lb = create_feeddata(dataset)
    sz = len(bs) // params['batch_size'] * params['batch_size']
    ts = ts[:sz]
    bs = bs[:sz]
    lb = lb[:sz]

    proba = model.predict_proba(ts, verbose=0, batch_size=params['batch_size'])
    out = np.hstack([proba, lb])
    sortout = out[(-out[:, proba.shape[proba.ndim - 1] - 1]).argsort(), :]

    if not __debug__:
        dt = datetime.date.today().strftime('%Y-%m-%d')
        np.savetxt(path+"/" + dt + "_result.txt", sortout, fmt='%f')
    else:
        print (sortout[0:20, :])

    idx = params['outdim'] - 1
    sortout = sortout[sortout[:, idx] >= 0.5][:, (idx + 2, -2, idx)]
    candidates = pd.DataFrame(sortout, columns=('code', 'price', 'proba'))
    print ("[ End prediction ] of tomorrow's price")
    return candidates


def predict_today2(mstrs, runs,force_return=False):
    print ("[ select stocks ]... using model:" + str(mstrs))
    index = None
    index = ['sme', 'gem']
    if __debug__:
        index = ['sme']
        pass

    models_days={}
    for i in range (len(mstrs)):
        params = parse_params(mstrs[i])
        models_days[mstrs[i]] = params['lookback']
    days=list(set(models_days.values()))
    dataset = ncreate_today_dataset_threads(index=index, days=days,force_return=force_return)

    candidates={}
    for i in range(len(mstrs)):
        params = parse_params(mstrs[i])
        path = 'models/' + params['mclass'] + '/' + params['model_name']
        if runs[i] is not None:
            path = path + '/' + runs[i]
        model = load_model(str(path) + '/best_model.h5', custom_objects=params['cmetrics'])
        print ("model summary:" + mstrs[i])
        model.summary()

        bs, ts, lb = create_feeddata(dataset[models_days[mstrs[i]]],copies=1)
        sz = len(bs) // params['batch_size'] * params['batch_size']
        ts = ts[:sz]
        bs = bs[:sz]
        lb = lb[:sz]

        proba = model.predict_proba(ts, verbose=0, batch_size=params['batch_size'])
        out = np.hstack([proba, lb])
        sortout = out[(-out[:, proba.shape[proba.ndim - 1] - 1]).argsort(), :]

        if not __debug__:
            dt = datetime.date.today().strftime('%Y-%m-%d')
            np.savetxt(path+"/" + dt + "_result.txt", sortout, fmt='%f')
        else:
            print (sortout[0:20, :])

        idx = params['outdim'] - 1
        sortout = sortout[sortout[:, idx] >= 0.5][:, (idx + 2, -2, idx)]

        candidates[mstrs[i]] = pd.DataFrame(sortout, columns=('code', 'price', 'proba'))
        candidates[mstrs[i]]['model'] = str(mstrs[i])

    print ("[ End prediction ] of tomorrow's price")
    cands = pd.concat(candidates.values())
    return cands


def predict_today_rpc(mstr, run, force_return=True):
    df = predict_today(mstr, run, force_return)
    return df.to_json()
    pass


def predict_today_rpc2(jsonstr, force_return=False):
    map = json.loads(jsonstr)
    df = predict_today2(map.keys(), map.values(), force_return)
    return df.to_json(orient='records')


def predict_today_rpc2_test(jsonstr, force_return=False):
    jsonstr = '{"MR_T2_B256_C2_E1000_Lclose_K5_XSGD": "S2016-01-01.20181009.11.29.52.run"' \
              ',"MR_T5_B256_C2_E2000_Lclose_K5_XSGD": "S2015-01-03.20181114.11.21.34.run"' \
              ',"MR_T10_B256_C2_E2000_Lclose_K5_XSGD":"S2015-01-03.20181119.17.27.09.run"}'
    force_return = False
    map = json.loads(jsonstr)
    df = predict_today2(map.keys(), map.values(), force_return)
    return df.to_json()



def print_model(mstr):
    params = parse_params(mstr)
    model = load_model('./models/' + params['model_name'] + '/best_model.h5', custom_objects=params['cmetrics'])
    print ("model summary")
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


def get_stats(group):
    return {'min': group.min(), 'max': group.max(), 'count': group.count(), 'mean': group.mean()}


def print_dist(proba, sortcol, labelcol, b=10.0):
    # print mean p_change of the proba result
    bins = np.arange(0, 1, 1 / float(b))
    labels = bins.searchsorted(proba[:, sortcol])
    grouped = pd.Series(proba[:, labelcol]).groupby(labels)
    print (grouped.apply(get_stats).unstack())
    if dir is not None:
        pass
        # pd.DataFrame(grouped.apply(get_stats).unstack()).to_csv("./models/" + dir + "/dist.txt")
    # end print


def print_dist_cut(proba, sortcol, labelcol, b=10.0, dir=None):
    # print mean p_change of the proba result
    factor = pd.cut(proba[:, sortcol], b)
    grouped = pd.Series(proba[:, labelcol]).groupby(factor)
    print (grouped.apply(get_stats).unstack())
    if dir is not None:
        # pd.DataFrame(grouped.apply(get_stats).unstack()).to_csv(dir + "/dist.txt")
        pass
    # end print


def start_service():
    TCPServer.request_queue_size = 10

    # create server
    server = SimpleXMLRPCServer(('127.0.0.1', 8080), SimpleXMLRPCRequestHandler, True)
    server.register_function(predict_today_rpc2, "predict_today_rpc2")
    server.serve_forever()
    # _main_()
    # predict_today('MR_T2_B256_C2_E1000_Lclose_K5_XSGD', 'S2016-01-01.20181009.11.29.52.run')
    # M1T5C3_D1()
    # predict_today()


def _main_():
    if len(sys.argv) > 7:
        eval(sys.argv[1])(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
    elif len(sys.argv) > 6:
        eval(sys.argv[1])(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    elif len(sys.argv) > 5:
        eval(sys.argv[1])(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif len(sys.argv) > 4:
        eval(sys.argv[1])(sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) > 3:
        eval(sys.argv[1])(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 2:
        eval(sys.argv[1])(sys.argv[2])
    else:
        eval(sys.argv[1])()
    exit(0)


if __name__ == '__main__':
    _main_()
    # M1T5C3_D1()
    # predict_today()


