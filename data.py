# coding=utf-8

# The DataManager class define operations of data management
# refresh_data([symbols]) : get fresh daily data from web and store locally
# get_daily_data([symbols], start, end, local): get data from local or web
# get_basic_date([symbols])
# get_current_data([symbols])
# get_finance_data([synbols])


import datetime
import sys
import threading
import time
import traceback
from datetime import timedelta

import pandas as pd
import tushare as ts
from pandas.io import sql
from sklearn import preprocessing
from sqlalchemy import create_engine

from data_utils import *

st_cat = {'sme': '399005', 'gem': '399006', 'hs300s': '000300', 'sz50s': '000016', 'zz500s': '000008'}
cons = ts.get_apis()
engine = create_engine('mysql://root:root@127.0.0.1/tushare?charset=utf8')
pd.set_option('mode.use_inf_as_na', True)


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


def refresh_kdata(ktype='5'):
    # refresh history data using get_k_data
    # force, force to get_k_data online

    print ("[ refresh_kdata ]... to k5_data table")

    data_table = "k" + ktype + "_data"
    tmp_table = "k" + ktype + "_tmp"
    merge_sql = "replace into " + data_table + \
                "(stmp, open, close, high, low, vol,amt, tor,vr) " \
                "select `date`, open, close, high, low, volume, turnoverratio " \
                "from " + tmp_table
    trunc_sql = "drop table if exists " + tmp_table

    sql.execute(trunc_sql, engine)

    # 获取股票K线数据
    basics = ts.get_stock_basics()
    basics.to_csv('./data/basics.csv')
    symbols = list(basics.index)

    # if __debug__:
    #     symbols = ['000506','600108']

    failed_symbols = []
    for symb in symbols:
        try:
            # 初始化
            merge_sql = "replace into " + data_table + \
                        "(stmp, code, open, close, high, low, vol, amt, tor, vr) " \
                        "select `datetime`,code, open, close, high, low, vol, amount, tor, vr " \
                        "from " + tmp_table
            df = pd.read_csv('./data/k5_data/' + symb + '.csv', dtype={'code': str})

            # df = ts.get_k_data(code=symb, start='2010-01-01', end='', ktype=ktype, autype='qfq')
            if df is not None and len(df) > 0:
                df.amount = 0
                df.to_sql(tmp_table, engine, if_exists='append', index=False)
        except:
            print ("Exception when processing stock:" + symb)
            traceback.print_exc()
            failed_symbols.append(symb)
            continue

    sql.execute(merge_sql, engine)
    print ("Failed stock symbols: ")
    print (failed_symbols)
    print ("[ end append_data ]")


def create_dataset_from_db(index=None, step=3, start=None, end=None, ktype='5'):
    step = int(step)
    print ("[ create_dataset]... of stock category %s with previous %i days" % (index, step))

    symbols = []
    basics = pd.read_csv('./data/basics.csv', index_col=0, dtype={'code': str})
    if index is None or len(index) == 0:
        symbols = int2str(list(basics.index))
    else:
        all = list(basics.index)
        if 'basic' in index:
            symbols.extend([x for x in all if x >= 600000])
        if 'sme' in index:
            symbols.extend([x for x in all if 300000 <= x < 600000])
        if 'gem' in index:
            symbols.extend([x for x in all if 300000 > x])
        symbols = int2str(symbols)
        if 'debug' in index:
            debug_df = pd.read_csv("./data/debug.csv", index_col=0, dtype={'code': str})
            symbols.extend(list(debug_df.code))
        if 'test' in index:
            debug_df = pd.read_csv("./data/test.csv", index_col=0, dtype={'code': str})
            symbols.extend(list(debug_df.code))

    sdate = datetime.datetime.strptime(start, '%Y-%m-%d')
    end = datetime.datetime.strptime(end, '%Y-%m-%d')
    start = next_n_busday(sdate, -step - 1).strftime('%Y-%m-%d')
    end = next_n_busday(end, 3).strftime('%Y-%m-%d')

    datesql = "'"
    if start is not None:
        datesql = datesql + " and stmp >= '" + start + "'"
    if end is not None:
        datesql = datesql + " and stmp <= '" + end + "'"

    data_all = []
    for symb in symbols:
        sql = "select DATE_FORMAT(stmp,'%%Y-%%m-%%d %%T') as stmp, code, open, close, high, low, vol, amt, tor, vr " \
              "from k5_data where code='" + symb + datesql + " order by stmp desc"
        # sql = 'select DATE_FORMAT(stmp,"%%Y-%%m-%%d %%T") from k5_data'
        df = pd.read_sql_query(sql=sql, con=engine, index_col='stmp')
        data_cells = create_cell_data(symb, df, ktype, step, start, end)
        data_all.extend(data_cells)

    print ("[ Finish create data set]")
    return data_all


def create_dataset_from_cvs(index=None, step=3, start=None, end=None, ktype='5'):
    step = int(step)
    print ("[ ncreate_dataset_from_cvs]... of stock category %s with previous %i days" % (index, step))

    sdate = datetime.datetime.strptime(start, '%Y-%m-%d')
    end = datetime.datetime.strptime(end, '%Y-%m-%d')
    start = next_n_busday(sdate, -step - 1).strftime('%Y-%m-%d')
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

    data_all = []
    for symb in symbols:
        df = pd.read_csv(path + symb + '.csv', index_col='datetime', dtype={'code': str})
        data_cells = create_cell_data(symb, df, ktype, step, start, end)
        data_all.extend(data_cells)

    print ("[ Finish create data set]")
    return data_all


def create_cell_data(symb, df, ktype, step, start, end):
    """

    :rtype: a list of data_cells
    """
    features = ['open', 'close', 'high', 'low', 'vol']
    knum = 240 // int(ktype)
    cells = []

    if df is not None and len(df) > 0:
        df = df[end:start]
        df = df.iloc[0:int(len(df) // knum * knum)]
        df.fillna(method='bfill')
        df.fillna(method='ffill')
        df.fillna(value=0)
        df.ix[:, 'vr'].fillna(value=1.0)
    else:
        return cells

    dclose = np.array(df.ix[-knum::-knum, 'close'])
    ddate = df.index[-knum::-knum]
    datall = np.array(df.ix[::-1, features])

    # 构建训练数据,nowcell为输入数据，max_price\min_price|cls_price|c2o_price为候选标签数据

    for i in range(1, len(df) // knum - step - 1):
        nowcell = np.array(datall[i * knum:(i + step) * knum])

        # 当天涨停，无法进行买入操作，删除此类案例
        if (dclose[i + step - 1] - dclose[i + step - 2]) / dclose[i + step - 2] > 0.099:
            continue

        # nowcell里的最后一个收盘价
        nowclose = nowcell[-1, 1]
        nxt2close = datall[(i + step + 2) * knum - 1, 1]

        nxtcell = np.array(datall[(i + step) * knum:(i + step + 1) * knum])
        max_price = min((max(nxtcell[:, 2]) - nowclose) / nowclose * 100, 10)
        min_price = max((min(nxtcell[:, 3]) - nowclose) / nowclose * 100, -10)
        cls_price = max(min((nxtcell[-1, 1] - nowclose) / nowclose * 100, 10), -10)
        c2o_price = max(min((nxtcell[0, 0] - nowclose) / nowclose * 100, 10), -10)
        cls2price = (nxt2close - nowclose) / nowclose * 100

        # # 把价格转化为变化的百分比*10, 数据范围为[-days,+days]，dclose[i-1]为上一个交易日的收盘价
        # nowcell[:,0:4] = (nowcell[:,0:4] - dclose[i-1]) / dclose[i-1] * 10# + K.epsilon()

        # 把价格转化为变化的百分比*10, 数据范围为[-1,+1]，dclose[i-1]为上一个交易日的收盘价
        for k in range(step):
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
        nowcell[-1, :] = nowcell[-2, :]

        if (abs(max_price) > 11 or abs(min_price) > 11) and __debug__:
            print ('*' * 50)
            print (lbdata)
            print ('*' * 50)
            continue

        bsdata = np.array(int2date(str2date(ddate[i + step].split(' ')[0])))

        # lbadata [日期，股票代码，最低价，最高价,pchange_days, c2o, cls, min, max]
        lbdata = [int2date(str2date(ddate[i + step].split(' ')[0])), int(symb), min(nxtcell[:, 3]), max(nxtcell[:, 2]), pchange_days, c2o_price, cls_price, min_price, max_price]

        cells.append([bsdata, nowcell, np.array(lbdata)])

    return cells


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


def create_today_dataset(index=None, days=[3,5], ktype='5', force_return=False):
    print ("[ create_dataset]... of stock category %s with previous %s days" % (index, str(days)))
    start_time = datetime.datetime.now()
    print (start_time)

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

    debug_count = 0
    for symb in symbols:
        debug_count = debug_count + 1
        if __debug__ and debug_count > 260:
            break

        # 超过下午2点58，立即返回，以便后续进行买卖操作
        if force_return and datetime.datetime.now().time() > datetime.time(14,59,0):
            break;

        try:
            start = (datetime.date.today() - timedelta(days=max(days) + 12)).strftime('%Y-%m-%d')
            dff = ts.get_k_data(code=symb, start=start, end='', ktype='5', autype='qfq')
            for d in days:
                cell = create_today_cell(symb, dff, d, ktype)
                if cell is not None:
                    data_all.get(d).append(cell)
        except:
            print ("Exception when processing index:" + symb)
            traceback.print_exc()
            continue

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


def create_today_dataset_threads(index=None, days=[3,5], ktype='5', force_return=False):
    print ("[ create_dataset]... of stock category %s with previous %s days" % (index, str(days)))
    start_time = datetime.datetime.now()
    print (start_time)

    global count, data_all

    for d in days:
        data_all[d] = []

    symbols = []
    if index is None or len(index) == 0:
        basics = get_basic_data()
        symbols = int2str(list(basics.index))
    else:
        for i in index:
            symbols.extend(list(get_index_list(i).code))
    st_symbols = list(ts.get_st_classified().code)
    symbols = [i for i in symbols if i not in st_symbols]

    threads = 4
    step = len(symbols) // threads
    symbol_list = [symbols[i:i+step] for i in xrange(0, len(symbols), step)]

    start = (datetime.date.today() - timedelta(days=max(days) + 12)).strftime('%Y-%m-%d')
    for symbs in symbol_list:
        t = threading.Thread(target=create_today_dataset_thread, args=(symbs,start, days, ktype, force_return, ))
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


def create_today_dataset_thread(symbs, start=None, days=[3, 5], ktype='5', force_return=False):
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
        try:
            dff = ts.get_k_data(code=symb, start=start, end='', ktype='5', autype='qfq')
            for d in days:
                cell = create_today_cell(symb, dff, d, ktype)
                if cell is not None:
                    if mutex.acquire():
                        data_all.get(d).append(cell)
                        mutex.release()
        except:
            print ("Exception when processing index:" + symb)
            traceback.print_exc()
            continue


def create_today_cell(symb, dff, d, ktype):
    knum = 240 // int(ktype)
    start = (datetime.date.today() - timedelta(days=d + 12)).strftime('%Y-%m-%d')
    features = ['open', 'close', 'high', 'low', 'volume']  # , 'vr']

    if dff is None or len(dff) <= knum * d:
        return None

    df = dff[dff['date'] > start]
    df = df.set_index('date')

    # 近日复牌或停牌数据，跳过
    if len(df) < knum * d:
        return None

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
        return None

    df.fillna(method='bfill')
    df.fillna(method='ffill')
    nowcell = np.array(df.ix[- d * knum:, features])

    # 把价格转化为变化的百分比*10, 数据范围为[-1,+1]，dclose[i-1]为上一个交易日的收盘价
    for k in range(d):
        nowcell[k * knum:(k + 1) * knum, 0:4] = (nowcell[k * knum:(k + 1) * knum, 0:4] - dclose[k]) / dclose[k] * 10 + K.epsilon()

    # 异常数据，跳过
    if abs(nowcell[:, 0:4].any()) > 1.1:
        return None

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
    ldata = [int2date(str2date(ddate[d].split(' ')[0])), int(symb), open, high, close, low]

    data_cell = [bsdata, nowcell, np.array(ldata)]

    return data_cell


def main():
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


if __name__ == '__main__':
    main()
    exit()
