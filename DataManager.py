# coding=utf-8

# The DataManager class define operations of data management
# refresh_data([symbols]) : get fresh daily data from web and store locally
# get_daily_data([symbols], start, end, local): get data from local or web
# get_basic_date([symbols])
# get_current_data([symbols])
# get_finance_data([synbols])


import os
import time
import traceback

import datetime
import h5py
import matplotlib as plt
import numpy as np
import pandas as pd
import tushare as ts
from datetime import date
from datetime import timedelta
from sklearn import preprocessing
import keras.backend as K

from keras.utils import np_utils

ffeatures = ['pe', 'outstanding', 'totals', 'totalAssets', 'liquidAssets', 'fixedAssets', 'reserved',
             'reservedPerShare', 'esp', 'bvps', 'pb', 'undp', 'perundp', 'rev', 'profit', 'gpr',
             'npr', 'holders']
bfeatures = ['pe', 'outstanding', 'reservedPerShare', 'esp', 'bvps', 'pb', 'perundp', 'rev', 'profit',
             'gpr', 'npr']
tsfeatures = ['open', 'high', 'close', 'low', 'p_change', 'turnover']
tfeatures = ['open', 'high', 'close', 'low', 'p_change']  # , 'volume']


def mydate(datestr):
    if isinstance(datestr, list) or isinstance(datestr, np.ndarray):
        datelist = []
        for ds in datestr:
            datelist.append(mydate(ds))
        return datelist
    else:
        datearr = datestr.split('-')
        if len(datearr) !=3: raise "Wrong date string format " + datestr
        return date(int(datearr[0]), int(datearr[1]), int(datearr[2]))


def intdate(dt):
    if isinstance(dt, list) or isinstance(dt, np.ndarray):
        intdatelist = []
        for d in dt:
            intdatelist.append(intdate(d))
        return intdatelist
    else:
        return dt.year*10000+dt.month*100+dt.day


def intstr(ints):
    if isinstance(ints, list) or isinstance(ints, np.ndarray):
        intarr = []
        for i in ints:
            intarr.append(int(i))
        return intarr
    else:
        return int(ints)


def int2str(ints):
    sb = '000000'
    if isinstance(ints, list) or isinstance(ints, np.ndarray):
        lst = []
        for i in ints:
            lst.append(sb[0:6-len(str(int(i)))] + str(i))
        return lst
    else:
        return sb[0:6-len(str(int(ints)))] + str(int(ints))


def minmax_scale(arr):
    mi = np.min(arr)
    mx = np.max(arr)
    arr = (arr-mi)/(mx-mi+K.epsilon())
    return arr

def pricechange_scale(arr):
    mi = np.min(arr)
    arr = (arr-mi)/(mi+K.epsilon())*100
    return arr

def catf2(data):
    data_y = data.copy()
    data_y[data_y < 1] = 31
    data_y[data_y < 31] = 32
    data_y -= 31
    data_y = np_utils.to_categorical(data_y, 2)
    return data_y

def catf3(data):
    data_y = data.copy()
    data_y[data_y < -1] = 11
    data_y[data_y < 1] = 12
    data_y[data_y < 10.5] = 13
    data_y -= 11
    data_y = np_utils.to_categorical(data_y, 3)
    return data_y

def catf31(data):
    data_y = data.copy()
    data_y[data_y < 0.5] = 31
    data_y[data_y < 3] = 32
    data_y[data_y < 30] = 33
    data_y -= 31
    data_y = np_utils.to_categorical(data_y, 3)
    return data_y

def catf4(data):
    data_y = data.copy()
    data_y[data_y < -1] = 11
    data_y[data_y < 1] = 12
    data_y[data_y < 3] = 13
    data_y[data_y <= 10.5] = 14
    data_y -= 11
    data_y = np_utils.to_categorical(data_y, 4)
    return data_y


def plot_out(sortout, x_index, y_index, points=200):
    step = len(sortout) / points
    plot_data = []
    i = 1
    plt.figure(1)
    while i * step < len(sortout):
        s = (i - 1) * step
        e = min(i * step, len(sortout))
        x = np.min(sortout[s:e, x_index])
        y = np.mean(sortout[s:e, y_index])
        plot_data.append([x, y])
        plt.plot(x, y, 'ro')
        i += 1
    plt.show()


def test_plot(mstr):
    d = np.loadtxt("./models/"+mstr+"/2017_02_2_result.txt")
    plot_out(d, 2, 3)


def refresh_data(start='2005-01-01', trytimes=10, force=False):
    # refresh history data
    # trytimes, times to try

    edate = datetime.date.today() - timedelta(days=1)
    edate = edate.strftime('%Y-%m-%d')
    data_path = './data/' + edate + '/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    elif not force:
        return
    print ("[ refresh_data ]... start date:%s" % (start))
    basics = ts.get_stock_basics()
    basics.to_csv('./data/basics.csv')
    symbols = list(basics.index)

    def trymore(symbs, times):
        failsymbs = []
        i = 0
        while i < len(symbs):
            try:
                df = ts.get_k_data(symbs[i], start,edate)
            except:
                failsymbs.append(symbs[i])
                print "Exception when processing " + symbs[i]
                traceback.print_exc()
                i += 1
                continue

            if df is not None and len(df) > 0:
                df = df[::-1]
                outstanding = basics.loc[symbs[i]]['outstanding'] * 100000000
                df['p_change'] = df['close'].diff()/df['close'][1:] * 100
                df['turnover'] = df['volume']/outstanding * 100
                df.to_csv(data_path + symbs[i] + '.csv')
                df.to_csv('./data/daily/' + symbs[i] + '.csv')
            else:
                failsymbs.append(symbs[i])
            i += 1
        if len(failsymbs) > 0:
            print "In round " + str(times) + " following symbols can't be resolved:\n" + str(failsymbs)
            if times - 1 > 0:
                times -= 1
                trymore(failsymbs, times)
            else:
                return
    trymore(symbols, trytimes)
    print ("[ end refresh_data ]")


def get_basic_data(online=False, cache=True):
    if online is False:
        basics = pd.read_csv('./data/basics.csv', index_col=0, dtype={'code': str})
    else:
        basics = ts.get_stock_basics()
        if cache is True and basics is not None:
            basics.to_csv('./data/basics.csv')
    return basics


def get_history_data(symb_num, totals=None, start=None, end=None):
    print ("[ get history data ]... for %i symbols" % (symb_num))
    # if symbols is None: return
    # refresh_data()
    basics = get_basic_data()
    symbols = int2str(list(basics.index))
    data_dict = {}
    i = 0
    while i < len(symbols) and i < symb_num:
        # 小市值股票
        if totals is not None and basics.iloc[i]['totals'] > totals: i += 1; continue
        try:
            df = pd.read_csv('./data/daily/' + symbols[i] + '.csv', index_col='date', dtype={'code': str})
            if df is not None:
                df = df[::-1]
                data_dict[symbols[i]] = df.loc[start:end][1:]
        except:
            if __debug__:
                print "Can't get data for symbol:" + str(symbols[i])
            else:
                pass
            # traceback.print_exc()
        i += 1
    print ("[ End get history data ]")
    return data_dict


def get_newly_data(days=300):
    print ("[ get newly data]... for %s days"%(str(days)))

    start = (datetime.date.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    end = (datetime.date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    today_file = './data/' + end + '/newlydata.h5'
    if os.path.exists(today_file):
        f = h5py.File(today_file, 'r')
        return f

    basics = ts.get_stock_basics()
    symbols = int2str(list(basics.index))
    f = h5py.File(today_file, 'w')
    i = 0
    while i < len(symbols):
        try:
            df = ts.get_h_data(symbols[i], start, end)
        except:
            traceback.print_exc()
            i += 1
            continue

        if df is not None and len(df) > 0:
            outstanding = basics.loc[symbols[i]]['outstanding'] * 100000000
            df = df[::-1]
            df['p_change'] = df['close'].diff() / df['close'][1:] * 100
            df['turnover'] = df['volume'] / outstanding * 100
            df = df[tsfeatures]
            f.create_dataset(symbols[i], data=np.array(df[1:].astype(float)))
        i += 1
    f.flush()
    return f


def get_newly_data2(start='2005-01-01', trytimes=10):
    # refresh history data
    # trytimes, times to try

    print ("[ get newly data]... from day %s"%(str(start)))

    end = (datetime.date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    today_file = './data/newlydata_' + str(intdate(mydate(end))) + '.h5'
    if os.path.exists(today_file):
        f = h5py.File(today_file, 'r')
        return f

    basics = ts.get_stock_basics()
    if basics is None or len(basics) == 0:
        basics = pd.read_csv('./data/basics.csv', index_col=0, dtype={'code': str})
    else:
        basics.to_csv('./data/basics.csv')
    symbols = list(basics.index)
    f = h5py.File(today_file, 'w')
    def trymore(symbs, times):
        failsymbs = []
        i = 0
        while i < len(symbs):
            try:
                df = ts.get_hist_data(symbs[i], start)[::-1]
            except:
                failsymbs.append(symbs[i])
                traceback.print_exc()
                i += 1
                continue

            if df is not None and len(df) > 0:
                datecol = np.array(intdate(mydate(list(df.index)))).reshape(-1, 1)
                df['p_change'] = df['p_change'].clip(-10, 10)
                outstanding = basics.loc[symbs[i]]['outstanding'] * 100000000
                df['turnover'] = df['volume']/outstanding * 100
                f.create_dataset(symbols[i], data=np.hstack([datecol,np.array(df[tsfeatures].astype(float))]))
            else:
                failsymbs.append(symbs[i])
            i += 1
        if len(failsymbs) > 0:
            print "In round " + str(times) + " following symbols can't be resolved:\n" + str(failsymbs)
            if times - 1 > 0:
                times -= 1
                trymore(failsymbs, times)
            else:
                return
    trymore(symbols, trytimes)
    print ("[ end get newlydata ]")


def create_dataset(sym_num, lookback=5, start=None, end=None, totals=None):
    """
    The function takes two arguments: the `dataset`, which is a NumPy array that we want to convert into a dataset,
    and the `lookback`, which is the number of previous time steps to use as input variables
    to predict the next time period — in this case defaulted to 5.
    symbs
    lookback: number of previous time steps as int
    returns a list of data cells of format([np.array(bsdata), tsdata, rtdata, lbdata])
    """

    print ("[ create_dataset]... look_back:%s"%lookback)

    sdate = datetime.datetime.strptime(start,'%Y-%m-%d')
    start = (sdate - timedelta(days=lookback/5*2+lookback)).strftime('%Y-%m-%d')

    data_all = []
    bsset = get_basic_data()[bfeatures]
    bsset = bsset[bsset['pb'] > 0]
    symblist = intstr(list(bsset.index))
    bsset = preprocessing.scale(bsset)
    bsset = np.hstack([np.array(symblist).reshape(-1, 1), bsset])
    stockset = get_history_data(sym_num, totals, start, end)
    for symb in stockset:
        if int(symb) not in symblist: continue
        bsdata = bsset[bsset[:, 0] == int(symb)][0]  # sym,...

        data_stock = stockset[symb][tsfeatures]
        datelist = mydate(list(data_stock.index))
        datecol = np.array(intdate(datelist)).reshape(-1, 1)
        p_change = np.array(data_stock['p_change'].clip(-10, 10))

        ndata_stock = np.array(data_stock)
        for i in range(len(ndata_stock) - lookback - 2):
            if ndata_stock[i+lookback-1,-2] > 9.94:
                continue  # clean data un-operational
            dtcell = ndata_stock[i:(i + lookback)]
            ohcl = minmax_scale(dtcell[:, 0:4])
            pchange = p_change[i:(i + lookback)].reshape(-1,1)/10
            turnover = minmax_scale(dtcell[:,-1]).reshape(-1,1)
            # 应该直对testcase进行normalization
            tsdata = np.hstack([datecol[i:i+lookback], ohcl, pchange, turnover])+K.epsilon()
            lbdata = np.hstack([[int(symb)], datecol[i+lookback], p_change[i+lookback-1:i+lookback+3], sum(p_change[i+lookback:i+lookback+2])])
            data_cell = [bsdata, tsdata, lbdata]
            data_all.append(data_cell)
    print "[ Finish create data set]"
    return data_all


def create_val_dataset(start, end, lookback, totals=5):
    """
    The function takes the `lookback`, which is the number of previous
    time steps to use as input variables
    to predict the next time period — in this case defaulted to 5.
    lookback: number of previous time steps as int
    returns a list of data cells of format([bsdata, tsdata, lbdata])
    """

    print ("[ create validation dataset ]... ")
    tsdataset = []
    lbdataset = []
    basics = get_basic_data()
    tsdata_dict = get_newly_data2()
    start = intdate(mydate(start))
    end = intdate(mydate(end))
    for symb in tsdata_dict:
        if basics.loc[int(symb)]['totals'] > totals: continue
        data_stock = np.array(tsdata_dict[symb])
        data_stock = data_stock[np.logical_and(data_stock[:,0]<=end,start<=data_stock[:,0])]
        for i in range(len(data_stock) - lookback - 2):
            if data_stock[i+lookback-1,-2] > 9.96:
                continue  # clean data un-operational
            ohcl = minmax_scale(data_stock[i:(i + lookback),1:5])
            pchange = data_stock[i:(i + lookback),-2].reshape(-1,1)/10
            turnover = minmax_scale(data_stock[i:(i + lookback),-1]).reshape(-1,1)
            # 应该直对testcase进行normalization
            tsdata = np.hstack([data_stock[i:i+lookback,0].reshape(-1,1), ohcl, pchange, turnover])
            lbdata = np.hstack([[int(symb)], data_stock[i+lookback:i+lookback+3,-2], sum(data_stock[i+lookback:i+lookback+3,-2])])
            tsdataset.append(tsdata)
            lbdataset.append(lbdata)
    tsdataset = np.array(tsdataset)
    lbdataset = np.array(lbdataset)
    tsdata_dict.close()
    return tsdataset, lbdataset


def create_today_dataset(lookback=5):
    """
    The function takes the `lookback`, which is the number of previous
    time steps to use as input variables
    to predict the next time period — in this case defaulted to 5.
    lookback: number of previous time steps as int
    returns a list of data cells of format([bsdata, tsdata, lbdata])
    """
    rtlabels = ['code', 'open', 'high', 'trade', 'low', 'changepercent', 'turnoverratio']
    tsdataset = []
    rtdataset = []
    print ("[ create today's dataset ]... for price prediction")

    tsdata_dict = get_newly_data()
    rtdata_df = ts.get_today_all()[rtlabels]
    symbs = np.array(rtdata_df['code'])
    rtset = np.array(rtdata_df.astype(float))
    for symb in tsdata_dict:
        if int(symb) not in intstr(symbs): continue
        rtdata_v = rtset[rtset[:, 0] == int(symb)][0]
        if rtdata_v[-2] > 9.94 or rtdata_v[-1] == 0:
            continue

        tsdata_df = tsdata_dict[symb]
        if len(tsdata_df) >= lookback - 1:
            ndata_stock = np.array(tsdata_df[1-lookback:])
            ndata_stock = np.vstack([ndata_stock, rtdata_v[1:]])

            p_change = ndata_stock[:,-2].reshape(-1, 1) / 10
            turnover = minmax_scale(ndata_stock[:,-1].reshape(-1, 1))
            ohcl = minmax_scale(ndata_stock[:, 0:4])
            tsdata = np.hstack([ohcl, p_change, turnover])
            tsdataset.append(tsdata)
            rtdataset.append(rtdata_v)
    tsdataset = np.array(tsdataset)
    rtdataset = np.array(rtdataset)
    tsdata_dict.close()
    return tsdataset, rtdataset


def split_dataset(dataset, train_psize, batch_size=1, seed=None):
    """
    Splits dataset into training and test datasets. The last `lookback` rows in train dataset
    will be used as `lookback` for the test dataset.
    :param dataset: source dataset
    :param train_psize: specifies the percentage of train data within the whole dataset
    :return: tuple of training data and test dataset
    """
    if seed is None:
        seed = time.mktime(time.localtime())

    print "[ split_dateset ]... into train and test with seed:" + str(seed)
    np.random.seed(int(seed))
    np.random.shuffle(dataset)
    # only take effect for array, so need to convert to numpy.array before shuffle
    # 多维矩阵中，只对第一维（行）做打乱顺序操作
    train_size = (long(len(dataset) * train_psize) / batch_size) * batch_size
    test_size = (len(dataset) - train_size) / batch_size * batch_size
    train = dataset[:train_size]
    test = dataset[train_size: train_size + test_size]
    print('[ Finish ] train_dataset: {}, test_dataset: {}'.format(len(train), len(test)))
    return train, test


def create_feeddata(dataset):
    """
    Splits dataset into data and labels.
    :param dataset: source dataset, a list of data cell of [bsdata, tsdata, lbdata]
    :return: tuple of (bsdata, tsdata, lbdata)
    """
    print "[ create_feeddata]..."
    rows = [len(dataset)]
    bsdata = np.zeros(rows + list(dataset[0][0].shape))
    tsdata = np.zeros(rows + list(dataset[0][1].shape))
    lbdata_v = np.zeros(rows + list(dataset[0][2].shape))
    i = 0
    while i < len(dataset):
        bsdata[i] = dataset[i][0]
        tsdata[i] = dataset[i][1]
        lbdata_v[i] = dataset[i][2]
        i += 1
    print "[ end create_feeddata]..."
    return bsdata, tsdata, lbdata_v


def balance_data(data_y, data_x, data_x2=None):
    # 对于数据倾斜（数据类别不平衡），此函数对少数类的样本进行复制，以消除类别的不平衡
    a = np.sum(data_y,axis=0)
    print "Category distribution before balancing"
    print a
    b = np.max(a)/(a)
    c = long(np.sum(a * b))
    data_xx = np.zeros([c]+list(data_x.shape[1:]))
    data_yy = np.zeros([c]+list(data_y.shape[1:]))
    data_xx2 = None
    if data_x2 is not None:
        data_xx2 = np.zeros([c] + list(data_x2.shape[1:]))
    l = 0
    for i in range(0, len(data_y)):
        t = b[np.argmax(data_y[i])]
        for j in range(0, int(t)):
            data_xx[l] = data_x[i]
            data_yy[l] = data_y[i]
            if data_x2 is not None:
                data_xx2[l] = data_x2[i]
            l += 1
    print "Category distribution after balancing"
    print np.sum(data_yy,axis=0)
    return data_yy, data_xx, data_xx2

def main():
    # data = create_dataset(['601866'])
    # print '#####data samples#############'
    # print data[0:2]

    # train, test = split_dataset(data, 0.7)
    # print '#####train samples#############'
    # print train[0:2]
    #
    # print '#####test samples##############'
    # print test[0:2]
    #
    # data_x, data_y = split_label(train)
    # print '#####train_x samples############'
    # print data_x[0:2]
    #
    # print '#####train_y samples############'
    # print data_y[0:2]

    # todata = get_todaydata(22, True, 10)
    # print '#####today data samples############'
    # print todata

    # arr = ['2010-01-10','2014-12-21','2014-1-29']
    # lst = np.array([u'2017-02-27', u'2017-02-24', u'2017-02-23', u'2017-02-22', u'2017-02-21'], dtype=object)
    # mydate(arr)
    # mydate(lst)

    # import copy
    # data = np.array(create_dataset2(['601866','600151','600152','600153'])[:3])
    # origdata = data.copy()
    # np.random.shuffle(data)
    # print '#####get dataset2 samples############'
    # print data
    refresh_data(force=True)

#    get_newly_data2('2016-01-01', 10)
#    pass

    # arr = np.arange(12).reshape(3,4)
    # print arr
    # a = minmax_scale(arr)
    # print a
    # print arr

if __name__ == '__main__':
    main()

"""
import DataManager as dm
dmr = dm.DataManager()
a,b,c = create_dataset(['601866'])
a,b = split_dataset(c,0.7)
"""
