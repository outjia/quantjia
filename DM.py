# coding=utf-8

from DataManager import *
from utils import *

cons = ts.get_apis()
engine = create_engine('mysql://root:root@127.0.0.1/tushare?charset=utf8')
pd.set_option('mode.use_inf_as_na', True)


def load_kdata(start='2010-01-01', ktype='5', force=False):
    # refresh history data using get_k_data
    # force, force to get_k_data online

    path = './data/k' + ktype + '_data/'

    print ("[ refresh_kdata ]... start date:%s in path %s" % (start, path))

    # 获取股票K线数据
    basics = ts.get_stock_basics()
    basics.to_csv('./data/basics.csv')
    failed_symbols = []
    for symb in list(basics.index):
        file = path + symb + '.csv'
        try:
            df = ts.bar(symb, conn=cons, adj='qfq', factors=['vr', 'tor'], freq=ktype + 'min', start_date=start, end_date='')
            if df is not None and len(df) > 0:
                df.to_csv(file)
                df.to_sql('k5_data', engine, if_exists='append')
        except:
            print ("Exception when processing stock:" + symb)
            traceback.print_exc()
            failed_symbols.append(symb)
            continue

    print ("Failed stock symbols: ")
    print (failed_symbols)
    print ("[ end refresh_data ]")


def ncreate_dataset_from_db(index=None, days=3, start=None, end=None, ktype='5'):
    print ("[ create_dataset]... of stock category %s with previous %i days" % (index, days))

    features = ['open', 'close', 'high', 'low', 'vol']

    sdate = datetime.datetime.strptime(start, '%Y-%m-%d')
    end = datetime.datetime.strptime(end, '%Y-%m-%d')
    start = next_n_busday(sdate, -days - 1).strftime('%Y-%m-%d')
    end = next_n_busday(end, 3).strftime('%Y-%m-%d')

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

    knum = 240 // int(ktype)

    datesql = "'"
    if start is not None:
        datesql = datesql + " and datetime >= '" + start + "'"
    if end is not None:
        datesql = datesql + " and datetime >= '" + end + "'"

    data_all = []
    for symb in symbols:
        sql = "select * from k5_data where code='" + symb + datesql + " order by datetime desc"
        try:
            df = pd.read_sql_query(sql=sql, con=engine, index_col='datetime')
            if df is not None and len(df) > 0:
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
            cls2price = (nxt2close - nowclose) / nowclose * 100

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

            if (abs(max_price) > 11 or abs(min_price) > 11) and __debug__:
                print ('*' * 50)
                print (lbdata)
                print ('*' * 50)
                continue

            bsdata = np.array(intdate(mydate(str(ddate[i + days]).split(' ')[0])))

            # lbadata [日期，股票代码，最低价，最高价,pchange_days, c2o, cls, min, max]
            lbdata = [intdate(mydate(str(ddate[i + days]).split(' ')[0])), int(symb), min(nxtcell[:, 3]), max(nxtcell[:, 2]), pchange_days, c2o_price, cls_price, min_price,
                      max_price]

            data_cell = [bsdata, nowcell, np.array(lbdata)]
            data_all.append(data_cell)
    print ("[ Finish create data set]")
    return data_all


def test():
    basics = pd.read_csv('./data/basics.csv', index_col=0, dtype={'code': str})
    sql = "select distinct code from k5_data"
    df = pd.read_sql_query(sql=sql, con=engine)
    b = int2str(list(basics.index))
    a = list(df['code'])
    symbols = [i for i in b if i not in a]
    fsyms = []
    t = 1
    while len(symbols) > 0 and t <= 3:
        for symb in symbols:
            try:
                # path='./data/k5_data/' + symb + '.csv'
                # if os.path.exists(path):
                #     pass
                # else:
                #     path = './data/k5_data_/' + symb + '.csv'
                df = pd.read_csv('./data/k5_data/' + symb + '.csv', index_col='datetime', dtype={'code': str})
                df.to_sql('k5_data', engine, if_exists='append')
            except:
                traceback.print_exc()
                fsyms.append(symb)
                continue
        print ("round: " + str(t))
        print str(fsyms)
        symbols = fsyms
        t = t + 1


def refresh():
    start = (datetime.date.today() - timedelta(days=10)).strftime('%Y-%m-%d')
    basics = pd.read_csv('./data/basics.csv', index_col=0, dtype={'code': str})
    symbols = int2str(list(basics.index))
    fsyms = []
    t = 1
    while len(symbols) > 0 and t <= 3:
        for symb in symbols:
            try:
                df = ts.get_k_data(symb, ktype='5', start=start)
                # df.rename(columns={'volume':'vol'}, inplace=True)
                df.to_sql('k5_data_tmp', engine, if_exists='append')
            except:
                traceback.print_exc()
                fsyms.append(symb)
                continue
        print ("refresh round: " + str(t))
        print str(fsyms)
        symbols = fsyms
        t = t + 1


def main():
    # load_kdata()
    test()
    # refresh()
    # db = ncreate_dataset_from_db(index='test', start='2017-08-10', end='2017-12-01')
    # df = ncreate_dataset(index='test', start='2017-08-10', end='2017-12-01')
    # print cmp(db, df)


if __name__ == '__main__':
    main()
    exit()

"""
import DataManager as dm
dmr = dm.DataManager()
a,b,c = create_dataset(['601866'])
a,b = split_dataset(c,0.7)
"""
