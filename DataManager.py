# coding=utf-8

# The DataManager class define operations of data management
# refresh_data([symbols]) : get fresh daily data from web and store locally
# get_daily_data([symbols], start, end, local): get data from local or web
# get_basic_date([symbols])
# get_current_data([symbols])
# get_finance_data([synbols])


import datetime
import os
import time
import traceback
from datetime import date
from datetime import timedelta

import keras.backend as K
import matplotlib as plt
import numpy as np
import pandas as pd
import tushare as ts
from business_calendar import Calendar
from keras.utils import np_utils
from sklearn import preprocessing

import threading

ffeatures = ['pe', 'outstanding', 'totals', 'totalAssets', 'liquidAssets', 'fixedAssets', 'reserved',
             'reservedPerShare', 'esp', 'bvps', 'pb', 'undp', 'perundp', 'rev', 'profit', 'gpr',
             'npr', 'holders']
bfeatures = ['pe', 'outstanding', 'reservedPerShare', 'esp', 'bvps', 'pb', 'perundp', 'rev', 'profit',
             'gpr', 'npr']
tsfeatures = ['open', 'high', 'close', 'low', 'p_change', 'turnover', 'idx_change']
tfeatures = ['open', 'high', 'close', 'low', 'p_change']  # , 'volume']

st_cat = {'sme': '399005', 'gem': '399006', 'hs300s': '000300', 'sz50s': '000016', 'zz500s': '000008'}
hdays = ['2013-01-01', '2013-01-02', '2013-01-03', '2013-05-01', '2013-05-02', '2013-05-03', '2013-10-01', '2013-10-02', '2013-10-03', '2013-10-04', '2013-10-05', '2013-10-06',
         '2013-10-07',
         '2014-01-01', '2014-01-02', '2014-01-03', '2014-05-01', '2014-05-02', '2014-05-03', '2014-10-01', '2014-10-02', '2014-10-03', '2014-10-04', '2014-10-05', '2014-10-06',
         '2014-10-07',
         '2015-01-01', '2015-01-02', '2015-01-03', '2015-05-01', '2015-05-02', '2015-05-03', '2015-10-01', '2015-10-02', '2015-10-03', '2015-10-04', '2015-10-05', '2015-10-06',
         '2015-10-07',
         '2016-01-01', '2016-01-02', '2016-01-03', '2016-05-01', '2016-05-02', '2016-05-03', '2016-10-01', '2016-10-02', '2016-10-03', '2016-10-04', '2016-10-05', '2016-10-06',
         '2016-10-07',
         '2017-01-01', '2017-01-02', '2017-01-03', '2017-05-01', '2017-05-02', '2017-05-03', '2017-10-01', '2017-10-02', '2017-10-03', '2017-10-04', '2017-10-05', '2017-10-06',
         '2017-10-07',
         '2017-01-01', '2018-01-02', '2018-01-03', '2018-05-01', '2018-05-02', '2018-05-03', '2018-10-01', '2018-10-02', '2018-10-03', '2018-10-04', '2018-10-05', '2018-10-06',
         '2018-10-07',
         '2019-01-01', '2019-01-02', '2019-01-03', '2019-05-01', '2019-05-02', '2019-05-03', '2019-10-01', '2019-10-02', '2019-10-03', '2019-10-04', '2019-10-05', '2019-10-06',
         '2019-10-07',
         '2020-01-01', '2020-01-02', '2020-01-03', '2020-05-01', '2020-05-02', '2020-05-03', '2020-10-01', '2020-10-02', '2020-10-03', '2020-10-04', '2020-10-05', '2020-10-06',
         '2020-10-07'
         ]


def next_n_busday(date, n):
    cal = Calendar(holidays=hdays)
    return cal.addbusdays(date, n)


def isbusday(date):
    cal = Calendar()
    return cal.isbusday(date)


def mydate(datestr):
    if isinstance(datestr, list) or isinstance(datestr, np.ndarray):
        datelist = []
        for ds in datestr:
            datelist.append(mydate(ds))
        return datelist
    else:
        datearr = datestr.split('-')
        if len(datearr) != 3: raise "Wrong date string format " + datestr
        return date(int(datearr[0]), int(datearr[1]), int(datearr[2]))


def intdate(dt):
    if isinstance(dt, list) or isinstance(dt, np.ndarray):
        intdatelist = []
        for d in dt:
            intdatelist.append(intdate(d))
        return intdatelist
    else:
        return dt.year * 10000 + dt.month * 100 + dt.day


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
            lst.append(sb[0:6 - len(str(int(i)))] + str(i))
        return lst
    else:
        return sb[0:6 - len(str(int(ints)))] + str(int(ints))


def minmax_scale(arr):
    mi = np.min(arr)
    mx = np.max(arr)
    arr = (arr - mi) / (mx - mi + K.epsilon()) + K.epsilon()
    return arr


def pricechange_scale(arr):
    mi = np.min(arr)
    arr = (arr - mi) / (mi + K.epsilon()) * 100
    return arr


def catf2(data):
    data_y = data.copy()
    data_y[data_y < 0.5] = 31
    data_y[data_y < 31] = 32
    data_y -= 31
    data_y = np_utils.to_categorical(data_y, 2)
    return data_y


def catf22(data):
    # 对low进行预测
    data_y = data.copy()
    data_y[data_y <= 1] = 31
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


def catf20(data):
    # 对low进行预测
    data_y = data.copy()
    data_y[data_y <= 0] = 31
    data_y[data_y < 31] = 32
    data_y -= 31
    data_y = np_utils.to_categorical(data_y, 2)
    return data_y


def catf31(data):
    data_y = data.copy()
    data_y[data_y < 0.1] = 31
    data_y[data_y < 3] = 32
    data_y[data_y < 30] = 33
    data_y -= 31
    data_y = np_utils.to_categorical(data_y, 3)
    return data_y


def noncatf(data):
    return data


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
    d = np.loadtxt("./models/" + mstr + "/2017_02_2_result.txt")
    plot_out(d, 2, 3)


def get_basic_data(online=False, cache=True):
    if online is False:
        basics = pd.read_csv('./data/basics.csv', index_col=0, dtype={'code': str})
    else:
        basics = ts.get_stock_basics()
        if cache is True and basics is not None:
            basics.to_csv('./data/basics.csv')
    return basics


def get_index_list(m):
    listfile = './data/' + m + '.csv'
    return pd.read_csv(listfile, index_col=0, dtype={'code': str})


def ncreate_dataset(index=None, days=3, start=None, end=None, ktype='5'):
    print ("[ create_dataset]... of stock category %s with previous %i days" % (index, days))

    features = ['open', 'close', 'high', 'low', 'vol']

    sdate = datetime.datetime.strptime(start, '%Y-%m-%d')
    end = datetime.datetime.strptime(end, '%Y-%m-%d')
    start = next_n_busday(sdate, -days - 1).strftime('%Y-%m-%d')
    end = next_n_busday(end, 3).strftime('%Y-%m-%d')
    path = './data/k' + ktype + '_data/'

    symbols = []
    if index is None or len(index) == 0:
        basics = get_basic_data()
        symbols = int2str(list(basics.index))
    else:
        basics = get_basic_data()
        all = list(basics.index)
        if 'basic' in index:
            symbols.extend([x for x in all if x >= 600000])
        if 'sme' in index:
            symbols.extend([x for x in all if 300000 <= x < 600000])
        if 'gem' in index:
            symbols.extend([x for x in all if 300000 > x])
        symbols = int2str(symbols)
        if 'debug' in index:
            symbols.extend(list(get_index_list('debug').code))
        if 'test' in index:
            debug_df = pd.read_csv("./data/test.csv", index_col=0, dtype={'code': str})
            symbols.extend(list(debug_df.code))

    knum = 240 // int(ktype)

    data_all = []
    for symb in symbols:
        try:
            df = pd.read_csv(path + symb + '.csv', index_col='datetime', dtype={'code': str})
            if df is not None and len(df) > 0:
                df = df[end:start]
                df = df.iloc[0:int(len(df) // knum * knum)]
                df.fillna(method='bfill')
                df.fillna(method='ffill')
                df.fillna(value=0)
                df.ix[:, 'vr'].fillna(value=1.0)
                if index is not None: df = df  # .join(idx_df)

            dclose = np.array(df.ix[-knum::-knum, 'close'])
            ddate = df.index[-knum::-knum]
            datall = np.array(df.ix[::-1, features])
        except:
            if __debug__:
                traceback.print_exc()
            else:
                # print "Can't get data for symbol:" + str(symb)
                pass
            continue

        # 构建训练数据,nowcell为输入数据，max_price\min_price|cls_price|c2o_price为候选标签数据
        for i in range(1, len(df) // knum - days - 1):
            nowcell = np.array(datall[i * knum:(i + days) * knum])

            # 当天涨停，无法进行买入操作，删除此类案例
            if (dclose[i + days - 1] - dclose[i + days - 2]) / dclose[i + days - 2] > 0.099:
                continue

            # nowcell里的最后一个收盘价
            nowclose = nowcell[-1, 1]
            nxt2close = datall[(i + days + 2) * knum - 1, 1]

            nxtcell = np.array(datall[(i + days) * knum:(i + days + 1) * knum])
            max_price = min((max(nxtcell[:, 2]) - nowclose) / nowclose * 100, 10)
            min_price = max((min(nxtcell[:, 3]) - nowclose) / nowclose * 100, -10)
            cls_price = max(min((nxtcell[-1, 1] - nowclose) / nowclose * 100, 10), -10)
            c2o_price = max(min((nxtcell[0, 0] - nowclose) / nowclose * 100, 10), -10)
            cls2price = max(min((nxt2close - nowclose) / nowclose * 100, 20), -20)

            # # 把价格转化为变化的百分比*10, 数据范围为[-days,+days]，dclose[i-1]为上一个交易日的收盘价
            # nowcell[:,0:4] = (nowcell[:,0:4] - dclose[i-1]) / dclose[i-1] * 10# + K.epsilon()

            # 把价格转化为变化的百分比*10, 数据范围为[-1,+1]，dclose[i-1]为上一个交易日的收盘价
            for k in range(days):
                nowcell[k * knum:(k + 1) * knum, 0:4] = (nowcell[k * knum:(k + 1) * knum, 0:4] - dclose[i + k - 1]) / dclose[i + k - 1] * 10 + K.epsilon()

            # 异常数据，跳过
            if abs(nowcell[:, 0:4].any()) > 1.1:
                continue

            # 过去days天股价变化总和，范围[-10*days, +10*days]
            pchange_days = cls2price

            try:
                j = 4
                if 'vol' in features:
                    # 归一化成交量
                    nowcell[:, j] = minmax_scale(preprocessing.scale(nowcell[:, j], copy=False))
                    j = j + 1
                if 'tor' in features:
                    # 归一化换手率
                    nowcell[:, j] = minmax_scale(nowcell[:, j])
                    j = j + 1
                if 'vr' in features:
                    # 归一化量比
                    nowcell[:, j] = minmax_scale(nowcell[:, j])
            except:
                if __debug__:
                    traceback.print_exc()
                pass

            # reshape to [days, knum, cols]
            # nowcell = nowcell.reshape(nowcell.shape[0] // knum, knum, nowcell.shape[-1])

            # 由于交易中无法获取当天真正的收盘价，而是以上一个k5数据作为最后一个5min的k线数据，所以修改测试数据和交易数据一致
            nowcell[-1,:] = nowcell[-2,:]

            if (abs(max_price) > 11 or abs(min_price) > 11) and __debug__:
                print ('*' * 50)
                print (lbdata)
                print ('*' * 50)
                continue

            bsdata = np.array(intdate(mydate(ddate[i + days].split(' ')[0])))

            # lbadata [日期，股票代码，最低价，最高价,pchange_days, c2o, cls, min, max]
            lbdata = [intdate(mydate(ddate[i + days].split(' ')[0])), int(symb), min(nxtcell[:, 3]), max(nxtcell[:, 2]), pchange_days, c2o_price, cls_price, min_price, max_price]

            data_cell = [bsdata, nowcell, np.array(lbdata)]
            data_all.append(data_cell)
    print ("[ Finish create data set]")
    return data_all


def ncreate_today_dataset2(index=None, days=[3,5], ktype='5', force_return=False):
    print ("[ create_dataset]... of stock category %s with previous %s days" % (index, str(days)))

    start_time = datetime.datetime.now()

    print (start_time)

    features = ['open', 'close', 'high', 'low', 'volume']  # , 'vr']

    day=max(days)
    start = (datetime.date.today() - timedelta(days=day + 12)).strftime('%Y-%m-%d')

    symbols = []
    if index is None or len(index) == 0:
        basics = get_basic_data()
        symbols = int2str(list(basics.index))
    else:
        for i in index:
            symbols.extend(list(get_index_list(i).code))
    #去除ST股票
    st_symbols = list(ts.get_st_classified().code)
    symbols = [i for i in symbols if i not in st_symbols]


    data_all = {}
    for d in days:
        data_all[d] = []

    knum = 240 // int(ktype)
    debug_count = 0
    for symb in symbols:

        debug_count = debug_count + 1
        if __debug__ and debug_count > 260:
            break

        # 超过下午2点58，立即返回，以便后续进行买卖操作
        if force_return and datetime.datetime.now().time() > datetime.time(14,59,0):
            break;

        try:
            dff = ts.get_k_data(code=symb, start=start, end='', ktype='5', autype='qfq')
        except:
            print ("Exception when processing index:" + symb)
            traceback.print_exc()
            continue

        for d in days:
            if dff is None or len(dff) <= knum * d:
                continue

            s = (datetime.date.today() - timedelta(days=day + 12)).strftime('%Y-%m-%d')
            df = dff[dff['date'] > s]

            # 近日复牌或停牌数据，跳过
            if len(df) <= knum * d:
                continue

            df = df.set_index('date')

            if len(df) == len(dff):
                #get_K_data返回14:55和15:00两个数据
                residual = len(df) % knum - 2
            else:
                residual = len(df) % knum

            if residual > 0:
                for i in range(knum - residual):
                    df = df.append(df.iloc[-1:])

            ddate = df.index[::knum]
            dclose = np.array(df.ix[(len(df) - 1)% knum::knum, 'close'])

            if(len(dclose)<=day):
                continue
            # 当天涨停，无法进行买入操作，删除此类案例
            if (dclose[d] - dclose[d - 1]) / dclose[d - 1] > 0.099:
                continue

            df.fillna(method='bfill')
            df.fillna(method='ffill')

            nowcell = np.array(df.ix[- d * knum:, features])

            # 把价格转化为变化的百分比*10, 数据范围为[-1,+1]，dclose[i-1]为上一个交易日的收盘价
            for k in range(d):
                nowcell[k * knum:(k + 1) * knum, 0:4] = (nowcell[k * knum:(k + 1) * knum, 0:4] - dclose[k]) / dclose[k] * 10 + K.epsilon()

            # 异常数据，跳过
            if abs(nowcell[:, 0:4].any()) > 1.1:
                continue

            try:
                j = 4
                if 'volume' in features:
                    # 归一化成交量
                    nowcell[:, j] = minmax_scale(preprocessing.scale(nowcell[:, j], copy=False))
                    j = j + 1
                if 'tor' in features:
                    # 归一化换手率
                    nowcell[:, j] = minmax_scale(nowcell[:, j])
                    j = j + 1
                if 'vr' in features:
                    # 归一化量比
                    nowcell[:, j] = minmax_scale(nowcell[:, j])
            except:
                pass

            bsdata = np.array(int(symb))
            high = float(max(df.ix[-48:, 'high']))
            open = float(df.ix[-48, 'open'])
            close = float(df.ix[-1, 'close'])
            low = float(min(df.ix[-48:, 'low']))
            # lbdata=[date, code, open, high, close, low]
            ldata = [intdate(mydate(ddate[d].split(' ')[0])), int(symb), open, high, close, low]

            data_cell = [bsdata, nowcell, np.array(ldata)]
            data_all.get(d).append(data_cell)

    #删除空的元组
    for d in days:
        if len(data_all.get(d))==0:
            data_all.pop(d)

    end_time = datetime.datetime.now()
    print ("[ Finish create data set] of " + str(debug_count) + " stocks, elapsed time:" + str(end_time - start_time))
    print (end_time)
    return data_all


count = 0
mutex = threading.Lock()
data_all = {}


def ncreate_today_dataset_threads(index=None, days=[3,5], ktype='5', force_return=False):
    print ("[ create_dataset]... of stock category %s with previous %s days" % (index, str(days)))
    global count, data_all
    start_time = datetime.datetime.now()
    print (start_time)
    day = max(days)

    start = (datetime.date.today() - timedelta(days=day + 12)).strftime('%Y-%m-%d')

    for d in days:
        data_all[d] = []

    count = 0
    threads = 4

    symbols = []
    if index is None or len(index) == 0:
        basics = get_basic_data()
        symbols = int2str(list(basics.index))
    else:
        for i in index:
            symbols.extend(list(get_index_list(i).code))
    st_symbols = list(ts.get_st_classified().code)
    symbols = [i for i in symbols if i not in st_symbols]

    step = len(symbols) // threads
    symbol_list = [symbols[i:i+step] for i in xrange(0, len(symbols), step)]

    for symbs in symbol_list:
        t = threading.Thread(target=ncreate_today_dataset_thread, args=(symbs,start, days, ktype, force_return, ))
        t.start()
        t.join()

    #删除空的元组
    for d in days:
        if len(data_all.get(d))==0:
            data_all.pop(d)

    end_time = datetime.datetime.now()
    print ("[ Finish create data set] of " + str(count) + " stocks, elapsed time:" + str(end_time - start_time))
    print (end_time)
    return data_all


def ncreate_today_dataset_thread(symbs, start=None, days=[3, 5], ktype='5', force_return=False):
    knum = 240 // int(ktype)
    features = ['open', 'close', 'high', 'low', 'volume']  # , 'vr']
    global count, mutex, data_all

    for symb in symbs:
        # if st_symbols.index(symb):continue
        # 超过下午2点58，立即返回，以便后续进行买卖操作
        if force_return and datetime.datetime.now().time() > datetime.time(14,59,0):
            break;

        if mutex.acquire():
            if __debug__ and count > 260:
                pass
            count = count + 1
            mutex.release()

        # time.sleep(0.005)
        try:
            dff = ts.get_k_data(code=symb, start=start, end='', ktype='5', autype='qfq')
        except:
            print ("Exception when processing index:" + symb)
            traceback.print_exc()
            continue

        for d in days:
            if dff is None or len(dff) <= knum * d:
                continue

            df = dff[dff['date'] > start]
            df = df.set_index('date')

            # 近日复牌或停牌数据，跳过
            if len(df) < knum * d:
                continue

            residual = len(df) % knum

            if len(df) == len(dff):
                residual = residual - 2

            if residual > 0:
                for i in range(knum - residual):
                    df = df.append(df.iloc[-1:])

            ddate = df.index[::knum]
            dclose = np.array(df.ix[len(df) % knum - 1::knum, 'close'])
            # 当天涨停，无法进行买入操作，删除此类案例
            if (dclose[d] - dclose[d - 1]) / dclose[d - 1] > 0.099:
                continue

            df.fillna(method='bfill')
            df.fillna(method='ffill')
            nowcell = np.array(df.ix[- d * knum:, features])

            # 把价格转化为变化的百分比*10, 数据范围为[-1,+1]，dclose[i-1]为上一个交易日的收盘价
            for k in range(d):
                nowcell[k * knum:(k + 1) * knum, 0:4] = (nowcell[k * knum:(k + 1) * knum, 0:4] - dclose[k]) / dclose[k] * 10 + K.epsilon()

            # 异常数据，跳过
            if abs(nowcell[:, 0:4].any()) > 1.1:
                continue

            try:
                j = 4
                if 'volume' in features:
                    # 归一化成交量
                    nowcell[:, j] = minmax_scale(preprocessing.scale(nowcell[:, j], copy=False))
                    j = j + 1
                if 'tor' in features:
                    # 归一化换手率
                    nowcell[:, j] = minmax_scale(nowcell[:, j])
                    j = j + 1
                if 'vr' in features:
                    # 归一化量比
                    nowcell[:, j] = minmax_scale(nowcell[:, j])
            except:
                pass

            bsdata = np.array(int(symb))
            high = float(max(df.ix[-48:, 'high']))
            open = float(df.ix[-48, 'open'])
            close = float(df.ix[-1, 'close'])
            low = float(min(df.ix[-48:, 'low']))
            # lbdata=[date, code, open, high, close, low]
            ldata = [intdate(mydate(ddate[d].split(' ')[0])), int(symb), open, high, close, low]

            data_cell = [bsdata, nowcell, np.array(ldata)]
            data_all.get(d).append(data_cell)

            if mutex.acquire():
                data_all.get(d).append(data_cell)
                mutex.release()


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

    print ("[ split_dateset ]... into train and test with seed:" + str(seed))
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


# def create_feeddata(dataset):
#     """
#     Splits dataset into data and labels.
#     :param dataset: source dataset, a list of data cell of [bsdata, tsdata, lbdata]
#     :return: tuple of (bsdata, tsdata, lbdata)
#     """
#     print ("[ create_feeddata]...")
#     rows = [len(dataset)]
#     if dataset[0][0] is not None:
#         bsdata = np.zeros(rows + list(dataset[0][0].shape))
#     else:
#         bsdata = np.zeros(rows)
#     tsdata = np.zeros(rows + list(dataset[0][1].shape))
#     lbdata_v = np.zeros(rows + list(dataset[0][2].shape))
#     i = 0
#     while i < len(dataset):
#         bsdata[i] = dataset[i][0]
#         tsdata[i] = dataset[i][1]
#         lbdata_v[i] = dataset[i][2]
#         i += 1
#     print ("[ end create_feeddata]...")
#     return bsdata, tsdata, lbdata_v


def create_feeddata(dataset, copies=1):
    """
    Splits dataset into data and labels.
    :param dataset: source dataset, a list of data cell of [bsdata, tsdata, lbdata];
    :param copies: copies of source dataset
    :return: tuple of (bsdata, tsdata, lbdata)
    """
    print ("[ create_feeddata]...")

    if len(dataset) == 0:
        return None, None, None

    data = dataset
    for j in range(copies-1):
        dataset.extend(data)
    np.random.shuffle(dataset)

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
    print ("[ end create_feeddata]...")
    return bsdata, tsdata, lbdata_v


def balance_data(data_y, data_x, data_x2=None):
    # 对于数据倾斜（数据类别不平衡），此函数对少数类的样本进行复制，以消除类别的不平衡
    a = np.sum(data_y, axis=0)
    print ("Category distribution before balancing")
    print (a)
    b = np.max(a) / (a)
    c = long(np.sum(a * b))
    data_xx = np.zeros([c] + list(data_x.shape[1:]))
    data_yy = np.zeros([c] + list(data_y.shape[1:]))
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
    print ("Category distribution after balancing")
    print (np.sum(data_yy, axis=0))
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
    ncreate_today_dataset_threads(index=['gem','sme'])
    # ncreate_today_dataset2(index=['gem'])
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
