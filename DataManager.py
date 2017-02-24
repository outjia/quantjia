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
                if data['high'][i] == data['low'][i]: continue # clean data of high equal to low
                # a = data[i:(i+look_back), :]
                # data_x.append(a)
                # data_y.append(tdata[i + look_back, :])
                data_all.append([ndata[i:(i+look_back), :],tndata[i + look_back, :]])
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

    def get_today_data(self, look_back):
        """
        Splits dataset into data and labels.
        :param dataset: source dataset, list of two elements
        :return: data_x of predication
        """
        sdate = None
        edate = None
        today_data = []

        if datetime.datetime.now().hour > 15:
            edate = datetime.date.today()
        else:
            edate = datetime.date.today() - timedelta(days=1)

        # in case of holidays without trading
        sdate = edate - timedelta(days = (look_back + 10))
        edate = edate.strftime('%Y-%m-%d')
        sdate = sdate.strftime('%Y-%m-%d')

        basics = ts.get_stock_basics()
        for symb in list(basics.index):
            try:
                data = ts.get_hist_data(symb, start=sdate, end=edate)
                if data is None: continue
            except:
                continue

            if len(data) < look_back or data['high'][-1] == data['low'][-1]: continue
            if data.index[0] != edate: continue

            data = data.drop(dfeatures, axis=1)

            for f in bfeatures:
                data[f] = basics.loc[symb][f]
            data['volume'] = data['volume']/10000

            #convert data to ndarray
            ndata = np.array(data)
            today_data.append([ndata[len(data)-look_back:len(data)]])
        print("Get latest %i stocks info from %i stocks.", len(today_data), len(basics))
        return today_data

def main():
    dmr = DataManager()
    data = dmr.create_dataset(['601866'])
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

    todata = dmr.get_today_data(5)
    print '#####today data samples############'
    np.savetxt("./data_new.txt", todata, fmt='%.f')
    print todata

if __name__ == '__main__':
    main()


"""
import DataManager as dm
dmr = dm.DataManager()
a,b,c = dmr.create_dataset(['601866'])
a,b = dmr.split_dataset(c,0.7)
"""