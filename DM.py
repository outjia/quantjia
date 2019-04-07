# coding=utf-8

from sqlalchemy import create_engine
from utils import *
import datetime
import traceback
from datetime import timedelta

import keras.backend as K
import numpy as np
import pandas as pd
import tushare as ts
from sklearn import preprocessing
from pandas.io import sql
import sys
import DataManager as dm
from data_utils import *

cons = ts.get_apis()
engine = create_engine('mysql://root:root@127.0.0.1/tushare?charset=utf8')
pd.set_option('mode.use_inf_as_na', True)


def refresh_kdata(ktype='5'):
    # refresh history data using get_k_data
    # force, force to get_k_data online

    print ("[ refresh_kdata ]... to k5_data table")

    data_table = "k"+ktype+"_data"
    tmp_table = "k" + ktype + "_tmp"
    merge_sql = "replace into "+data_table+ \
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
        data_cells = create_cell_data(symb, df,ktype,step,start,end)
        if data_cells is None:
            pass
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

        bsdata = np.array(intdate(mydate(ddate[i + step].split(' ')[0])))

        # lbadata [日期，股票代码，最低价，最高价,pchange_days, c2o, cls, min, max]
        lbdata = [intdate(mydate(ddate[i + step].split(' ')[0])), int(symb), min(nxtcell[:, 3]), max(nxtcell[:, 2]), pchange_days, c2o_price, cls_price, min_price, max_price]

        cells.append([bsdata, nowcell, np.array(lbdata)])

    return cells


def test_create_dataset():
    dataset_from_db = create_dataset_from_db('debug', 10, '2018-01-01', '2018-01-20')
    dataset_from_csv = dm.ncreate_dataset('debug', 10, '2018-01-01', '2018-01-20')

    e = equals_list(dataset_from_db, dataset_from_csv)

    a= str(dataset_from_db)
    b= str(dataset_from_csv)
    pass


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

"""
import DataManager as dm
dmr = dm.DataManager()
a,b,c = create_dataset(['601866'])
a,b = split_dataset(c,0.7)
"""
