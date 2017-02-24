# coding=utf-8

# The DataManager class define operations of data management
# refresh_data([symbols]) : get fresh daily data from web and store locally
# get_daily_data([symbols], start, end, local): get data from local or web
# get_basic_date([symbols])
# get_current_data([symbols])
# get_finance_data([synbols])


import ConfigParser
import os

import tushare as ts
import pandas as pd
import time
import csv
import traceback
import numpy as np
from keras.utils import np_utils, generic_utils
import keras as ks
import datetime
from datetime import timedelta

ffeatures = ['pe', 'outstanding', 'totals', 'totalAssets', 'liquidAssets', 'fixedAssets', 'reserved',
             'reservedPerShare', 'esp', 'bvps', 'pb', 'undp', 'perundp', 'rev', 'profit', 'gpr',
             'npr', 'holders']
bfeatures = ['pe', 'outstanding', 'reservedPerShare', 'esp', 'bvps', 'pb', 'perundp', 'rev', 'profit',
             'gpr', 'npr']
dfeatures = ['price_change', 'ma5', 'ma10', 'ma20', 'v_ma5', 'v_ma10', 'v_ma20']
tfeatures = ['p_change', 'open', 'high', 'close', 'low']  # , 'volume']


class DataManager():
    def __init__(self, cfgfile='config.cfg', data_path='data/', logfile='dm.log'):
        self.configParser = ConfigParser.ConfigParser()
        self.configParser.read(cfgfile)
        self.data_path = data_path
        self.storage = self.configParser.get('dbconfig', 'storage')
        self.logfile = logfile

    def refresh_data(self, symbols = None, data_type = 'H', trytimes=3):
        # refresh history data
        # data_type = 'H', refresh h_data
        # data_type = 'B', refresh basics
        # trytimes, times to try

        if trytimes == 0:
            return

        symbs = []

        if symbols is None:
            all_data = ts.get_today_all()
            symbols = all_data['code']

        if self.storage == 'csv':
            if 'B' in data_type:
                basics = ts.get_stock_basics()
                basics.to_csv(self.data_path+'basics.csv')

            if 'H' in data_type:
                i = 0
                while i < len(symbols):
                    try:
                        df = ts.get_hist_data(symbols[i])
                    except:
                        symbs.append(symbols[i])
                        print "Exception when processing " + symbols[i]
                        traceback.print_exc()
                        i = i + 1
                        continue

                    if df is not None:
                        df.to_csv(self.data_path+'daily/'+symbols[i]+'.csv')
                    else:
                        # TODO: add log trace
                        symbs.append(symbols[i])
                    i = i + 1
        else:
            # storage:database
            return

        if len(symbs) > 0:
            print "In round " + str(trytimes) + " following symbols can't be resolved:\n" + symbs
            self.refresh_data(symbs, 'H', trytimes - 1)

    # end of refresh_data



    def get_data(self, symbols=None,  start=None, end=None, data_type='H', online=False, cache=True):
        # get data
        # data_type = 'H', get h_data
        # data_type = 'B', get basics
        # online = False, get local data
        if online is False and self.storage == 'csv':
            if 'B' in data_type:
                return pd.read_csv(self.data_path+'basics.csv', index_col =0,dtype={'code':str})

            if 'H' in data_type and symbols is not None:
                i = 0
                dict = {}
                while i < len(symbols):
                    try:
                        dict[symbols[i]] = pd.read_csv(self.data_path+'daily/'+symbols[i]+'.csv', index_col =0,dtype={'code':str})
                    except:
                        print "Can't get data for symbol:" + str(symbols[i])
                    i = i + 1
                return dict

        elif online is True:
            if 'B' in data_type:
                basics = ts.get_stock_basics()
                if cache is True: basics.to_csv(self.data_path+'basics.csv')
                return basics

            if 'H' in data_type:
                i = 0
                dict = {}
                while i < len(symbols):
                    df = ts.get_hist_data(symbols[i])
                    dict[symbols[i]] = df
                    if cache is True: df.to_csv(self.data_path+'daily/'+symbols[i]+'.csv')
                    i = i + 1
                return dict
        else:
            # storage:database
            print 'Storage in database hasn''t been supported'
            return None
    # end of get data

    def norm_data(self, stkdata):
        pass


    def create_dataset(self, symbs, look_back=5):
        """
        The function takes two arguments: the `dataset`, which is a NumPy array that we want to convert into a dataset,
        and the `look_back`, which is the number of previous time steps to use as input variables
        to predict the next time period — in this case defaulted to 5.
        symbs
        look_back: number of previous time steps as int
        returns tuple of input and output dataset
        """
        # data_x, data_y = [], []
        data_all = []
        data_basic = self.get_data(data_type='B')
        data_stock = self.get_data(symbs)

        for symb in data_stock:
            data = data_stock[symb]
            data = data.drop(dfeatures, axis=1)
            for f in bfeatures:
                data[f] = data_basic.loc[int(symb)][f]
            data['volume'] = data['volume']/10000
            #convert data to ndarray
            ndata = np.array(data)
            tndata = np.array(data[tfeatures])
            for i in range(len(data)-look_back-1):
                if data['high'][i] == data['low'][i]:
                    continue # clean data of high equal to low
                data_all.append([ndata[i:(i+look_back), :],tndata[i + look_back, :]])
        return data_all

    def create_dataset2(self, symbs, look_back=5):
        """
        The function takes two arguments: the `dataset`, which is a NumPy array that we want to convert into a dataset,
        and the `look_back`, which is the number of previous time steps to use as input variables
        to predict the next time period — in this case defaulted to 5.
        symbs
        look_back: number of previous time steps as int
        returns a list of data cells
        """
        data_all = []

        data_basic = self.get_data(data_type='B')
        data_stocks = self.get_data(symbs)

        for symb in data_stocks:
            data_cell = []  # a data instance, for training and test
            bdata = [int(symb)]  # sym,...
            tsdata = []  # time serial data
            rtdata = []  # real time data, current opening price etc.
            ldata = []  # label data, day+1, day+2

            data_stock = data_stocks[symb]
            data_stock = data_stock.drop(dfeatures, axis=1)
            ldata_stock = []

            # basic data
            for f in bfeatures:
                bdata.append(data_basic.loc[int(symb)][f])

            data_stock['volume'] = data_stock['volume']/10000
            #convert data to ndarray
            ndata_stock = np.array(data_stock)
            ldata_stock = np.array(data_stock[tfeatures])
            for i in range(len(data_stock)-look_back-2):
                if data_stock['high'][i] == data_stock['low'][i]:
                    continue # clean data of high equal to low
                tsdata = ndata_stock[i:(i+look_back), :]
                rtdata = ldata_stock[i + look_back, :]
                ldata = ldata_stock[i + look_back + 1, :]
                data_cell = [np.array(bdata), tsdata, rtdata, ldata]
                data_all.append(data_cell)
        return data_all

    def split_dataset(self, dataset, train_psize, batch_size = 1):
        """
        Splits dataset into training and test datasets. The last `look_back` rows in train dataset
        will be used as `look_back` for the test dataset.
        :param dataset: source dataset
        :param train_psize: specifies the percentage of train data within the whole dataset
        :return: tuple of training data and test dataset
        """
        np.random.seed(19801016)
        np.random.shuffle(dataset)
        # only take effect for array, so need to convert to numpy.array before shuffle
        # 多维矩阵中，只对第一维（行）做打乱顺序操作
        train_size = (long(len(dataset) * train_psize) / batch_size) * batch_size
        test_size = (len(dataset)-train_size) / batch_size * batch_size
        train = dataset[0:train_size]
        test = dataset[train_size : train_size + test_size]
        print('train_dataset: {}, test_dataset: {}'.format(len(train), len(test)))
        return train, test

    def split_label(self, dataset):
        """
        Splits dataset into data and labels.
        :param dataset: source dataset, list of two elements
        :return: tuple of training data and test dataset
        """
        data_x, data_y = [], []
        for d in dataset:
            data_x.append(d[0])
            data_y.append(d[1])
        data_x = np.array(data_x)
        data_x = np.reshape(data_x, (data_x.shape[0], data_x.shape[1], data_x.shape[2])) #TODO can be removed?
        data_y = np.array(data_y)

        return data_x, data_y

    def catnorm_data(data_y):
        data_y[data_y < -2] = 11
        data_y[data_y < 2] = 12
        data_y[data_y < 11] = 13
        data_y = data_y - 11
        data_y = np_utils.to_categorical(data_y, 3)
        return data_y

    def get_todaydata(self, look_back, refresh=False, trytimes=3):
        """
        Splits dataset into data and labels.
        :param dataset: source dataset, list of two elements
        :return: data_x of predication
        """
        sdate = None
        edate = None
        today_data = []
        cachefile = None

        if datetime.datetime.now().hour > 15:
            edate = datetime.date.today()
        else:
            edate = datetime.date.today() - timedelta(days=1)
        cachefile = './data/todaydata' + edate.strftime('%Y%m%d') + '.npy'

        if refresh==False:
            try:
                today_data = np.load(cachefile)
                if today_data is not None and len(today_data) != 0:
                    return today_data
                else:
                    pass
            except:
                pass
        # in case of holidays without trading
        sdate = edate - timedelta(days = (look_back + look_back/5 * 2 + 20))
        edate = edate.strftime('%Y-%m-%d')
        sdate = sdate.strftime('%Y-%m-%d')
        basics = ts.get_stock_basics()

        def trymore(trytimes, symbs):
            if trytimes == 0 or symbs is None or len(symbs) == 0: return
            failedsymbs = []
            for symb in list(symbs):
                try:
                    data = ts.get_hist_data(symb, start=sdate, end=edate)
                    if data is None:
                        failedsymbs.append(symb)
                        continue
                except:
                    failedsymbs.append(symb)
                    continue

                if len(data) < look_back or data['high'][-1] == data['low'][-1]: continue
                if data.index[0] != edate: continue
                data = data.drop(dfeatures, axis=1)
                for f in bfeatures:
                    data[f] = basics.loc[symb][f]
                data['volume'] = data['volume'] / 10000

                # convert data to ndarray
                ndata = np.array(data)
                today_data.append(ndata[len(data) - look_back:len(data)])
            trymore(trytimes - 1, failedsymbs)
            return
        trymore(trytimes, basics.index)
        today_data = np.array(today_data)
        np.save(cachefile, today_data)
        print("Get latest %i stocks info from %i stocks." % (len(today_data), len(basics)))
        return today_data

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

def main():
    dmr = DataManager()
    # data = dmr.create_dataset(['601866'])
    # print '#####data samples#############'
    # print data[0:2]

    # train, test = dmr.split_dataset(data, 0.7)
    # print '#####train samples#############'
    # print train[0:2]
    #
    # print '#####test samples##############'
    # print test[0:2]
    #
    # data_x, data_y = dmr.split_label(train)
    # print '#####train_x samples############'
    # print data_x[0:2]
    #
    # print '#####train_y samples############'
    # print data_y[0:2]

    todata = dmr.get_todaydata(22, True, 10)
    print '#####today data samples############'
    print todata

    # import copy
    # data = np.array(dmr.create_dataset2(['601866','600151','600152','600153'])[:3])
    # origdata = data.copy()
    # np.random.shuffle(data)
    # print '#####get dataset2 samples############'
    # print data

if __name__ == '__main__':
    main()


"""
import DataManager as dm
dmr = dm.DataManager()
a,b,c = dmr.create_dataset(['601866'])
a,b = dmr.split_dataset(c,0.7)
"""