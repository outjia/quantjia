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
tsfeatures = ['open', 'high', 'close', 'low', 'p_change', 'turnover','idx_change']
tfeatures = ['open', 'high', 'close', 'low', 'p_change']  # , 'volume']

st_cat = {'sme':'399005','gem':'399006','hs300s':'000300', 'sz50s':'000016', 'zz500s':'000008'}

cons = ts.get_apis()


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


def intdate2str(dt):
    if isinstance(dt, list) or isinstance(dt, np.ndarray):
        strdate = []
        for d in dt:
            strdate.append(intdate2str(d))
        return strdate
    else:
        year = dt/10000
        month = (dt - year*10000)/100
        day = dt - year*10000 - month*100
        return str(year)+'-'+str(month)+'-'+str(day)


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
    arr = (arr-mi)/(mx-mi+K.epsilon())+K.epsilon()
    return arr

def pricechange_scale(arr):
    mi = np.min(arr)
    arr = (arr-mi)/(mi+K.epsilon())*100
    return arr

def catf2(data):
    data_y = data.copy()
    data_y[data_y < 0.1] = 31
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
    data_y[data_y < 0.5] = 51
    data_y[data_y < 2] = 52
    data_y[data_y < 50] = 53
    data_y -= 51
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


def catf21(data):
    # 对low进行预测
    data_y = data.copy()
    data_y[data_y <= 0] = 31
    data_y[data_y < 31] = 32
    data_y -= 31
    data_y = np_utils.to_categorical(data_y, 2)
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


def refresh_kdata(start='2015-01-01', ktype='5', force=False):
    # refresh history data using get_k_data
    # force, force to get_k_data online

    edate = datetime.date.today() - timedelta(days=1)
    edate = edate.strftime('%Y-%m-%d')
    path = './data/k'+ktype+'_data/'
    if not os.path.exists(path+edate):
        if not os.path.exists(path):
            os.mkdir(path)
        os.system('touch ' + path + edate)
    elif not force:
        print ("[ refresh_kdata ]... using existing data in "%(path))
        return

    print ("[ refresh_kdata ]... start date:%s in path %s" % (start, path))

    index = ts.get_index()
    basics = ts.get_stock_basics()
    basics.to_csv('./data/basics.csv')
    symbols = list(basics.index) + list(index.code)

    failsymbs = []
    for symb in symbols:
        file = path + symb + '.csv'
        if os.path.exists(file):
            continue

        try:
            df = ts.bar(symb,conn=cons,adj='qfq',factors=['vr','tor'],freq=ktype+'min',start_date=start,end_date='')
        except:
            print "Exception when processing " + symb
            failsymbs.append(symb)
            traceback.print_exc()
            continue
        if df is not None and len(df) > 0:
            df.to_csv(file)
    print "Failed Symbols: "
    print failsymbs

    print "获取指数成分股列表"
    clist = ts.get_sme_classified()
    if clist is not None and len(clist) > 0:
        clist.to_csv('./data/sme.csv')

    clist = ts.get_gem_classified()
    if clist is not None and len(clist) > 0:
        clist.to_csv('./data/gem.csv')

    clist = ts.get_hs300s()
    if clist is not None and len(clist) > 0:
        clist.to_csv('./data/hs300s.csv')

    clist = ts.get_sz50s()
    if clist is not None and len(clist) > 0:
        clist.to_csv('./data/sz50s.csv')

    clist = ts.get_zz500s()
    if clist is not None and len(clist) > 0:
        clist.to_csv('./data/zz500s.csv')

    print ("[ end refresh_data ]")


def get_basic_data(online=False, cache=True):
    if online is False:
        basics = pd.read_csv('./data/basics.csv', index_col=0, dtype={'code': str})
    else:
        basics = ts.get_stock_basics()
        if cache is True and basics is not None:
            basics.to_csv('./data/basics.csv')
    return basics


def get_index_list(m):
    listfile = './data/'+ m +'.csv'
    return pd.read_csv(listfile, index_col=0, dtype={'code': str})


def ncreate_dataset(index=None, days=3, start=None, end=None, ktype='5'):
    print ("[ create_dataset]... of stock category %s with %i" % (index, days))

    if __debug__:
        index = 'debug'

    sdate = datetime.datetime.strptime(start, '%Y-%m-%d')
    start = (sdate - timedelta(days=days / 5 * 2 + days)).strftime('%Y-%m-%d')
    path = './data/k' + ktype + '_data/'

    basics = get_basic_data()
    symbols = int2str(list(basics.index))
    if not (index is None or index == ''):
        symbols = list(get_index_list(index).code)
        # idx_df = pd.read_csv(path + st_cat[index] + '.csv', index_col='date', dtype={'code': str})

    knum = 240/int(ktype)

    data_all = []
    for symb in symbols:
        try:
            df = pd.read_csv(path + symb + '.csv', index_col='datetime', dtype={'code': str})
            if df is not None:
                df = df[end:start]
                df.fillna(method='bfill')
                df.fillna(method='ffill')
                if index is not None: df = df#.join(idx_df)

            dclose = np.array(df.ix[-knum::-knum,'close'])
            ddate = df.index[-knum::-knum]
            datall = np.array(df.ix[::-1,['open','close','high','low','vol']])

            # df = df[end:start]
        except:
            if __debug__:
                traceback.print_exc()
            else:
                # print "Can't get data for symbol:" + str(symb)
                pass
            continue

        for i in range(1, len(df)/knum-days):
            nowcell = np.array(datall[i*knum:(i+days)*knum])
            nxtcell = np.array(datall[(i+days)*knum:(i+days+1)*knum])
            for k in range(days):
                nowcell[k*knum:(k+1)*knum,0:4] = (nowcell[k*knum:(k+1)*knum,0:4] - dclose[i+k-1]) / dclose[i+k-1] * 10 + K.epsilon()

            # 异常数据，跳过
            if abs(nowcell[:,0:4].any()) > 11:continue
            # 当天涨停，删除此类案例
            if nowcell[-1,1] > 0.98: continue

            pchange_pre = float(nowcell[-1,1]*10)

            try:
                nowcell[:,4] = preprocessing.scale(nowcell[:,4],copy=False)
            except:
                pass
            nowcell = nowcell.reshape(nowcell.shape[0]/knum,knum,nowcell.shape[-1])

            max_price = min((max(nxtcell[:,2])-dclose[i+days-1])/dclose[i+days-1]*100, 20)
            min_price = max((min(nxtcell[:,3])-dclose[i+days-1])/dclose[i+days-1]*100,-20)
            cls_price = (nxtcell[-1,1]-dclose[i+days-1])/dclose[i+days-1] * 100
            lbdata = [intdate(mydate(ddate[i+days].split(' ')[0])),int(symb),min(nxtcell[:,3]),max(nxtcell[:,2]),pchange_pre, cls_price,min_price,max_price]

            if abs(max_price)>11 or abs(min_price) > 11:
                # print '*' * 50
                # print lbdata
                # print '*' * 50
                continue

            bsdata = np.array(intdate(mydate(ddate[i+days].split(' ')[0])))
            data_cell = [bsdata, nowcell[:,:,-3:], np.array(lbdata)]
            data_all.append(data_cell)
    print "[ Finish create data set]"
    return data_all


def get_newly_kdata(ktype='5',days=30, inc=True):
    print ("[ get newly kdata]... for %s days"%(str(days)))

    start = (datetime.date.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    end = (datetime.date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    today_file = './data/' + end + '/newlykdata.h5'
    if os.path.exists(today_file):
        f = h5py.File(today_file, 'r')
        return f
    f = h5py.File(today_file, 'w')

    failsymbs = []
    if inc and os.path.exists('./data/' + end + '/fails.csv'):
        symbols = pd.read_csv('./data/' + end + '/fails.csv',dtype='str')
    else:
        index = ts.get_index()
        basics = ts.get_stock_basics()
        basics.to_csv('./data/basics.csv')
        symbols = list(basics.index) + list(index.code)

    for symb in symbols:
        try:
            df = ts.bar(symb,conn=cons,adj='qfq',factors=['vr','tor'],freq=ktype+'min',start_date=start,end_date='')
        except:
            print "Exception when processing " + symb
            failsymbs.append(symb)
            traceback.print_exc()
            continue
        if df is not None and len(df) > 0:
            f.create_dataset(symb, data=df)
    f.flush()
    return f


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

    tsdata_dict = get_newly_kdata()
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
    if dataset[0][0] is not None:
        bsdata = np.zeros(rows + list(dataset[0][0].shape))
    else:
        bsdata = np.zeros(rows)
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
    refresh_kdata(force=True)
    # ncreate_dataset(start='2016-01-01')
#     get_history_data(4000, start='2005-01-01', end=None, index=None)
# #    get_newly_data2('2016-01-01', 10)
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
