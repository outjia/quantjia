# coding=utf-8


from datetime import date

import keras.backend as K
import matplotlib as plt
import numpy as np
from business_calendar import Calendar
from keras.utils import np_utils

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

cal = Calendar(holidays=hdays)


def next_n_busday(date, n):
    return cal.addbusdays(date, n)


def isbusday(date):
    cal = Calendar()
    return cal.isbusday(date)


def str2date(datestr):
    if isinstance(datestr, list) or isinstance(datestr, np.ndarray):
        datelist = []
        for ds in datestr:
            datelist.append(str2date(ds))
        return datelist
    else:
        datearr = datestr.split('-')
        if len(datearr) != 3: raise "Wrong date string format " + datestr
        return date(int(datearr[0]), int(datearr[1]), int(datearr[2]))


def int2date(dt):
    if isinstance(dt, list) or isinstance(dt, np.ndarray):
        intdatelist = []
        for d in dt:
            intdatelist.append(int2date(d))
        return intdatelist
    else:
        return dt.year * 10000 + dt.month * 100 + dt.day


def str2int(ints):
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


def equals_list(list1, list2):
    if isinstance(list1, list) and isinstance(list2, list):
        if len(list1) <> len(list2):
            return False
        for i in range(len(list1)):
            if list2[i] is not None:
                if not equals_list(list1[i], list2[i]):
                    return False
                else:
                    continue
            else:
                return False
    elif isinstance(list1, np.ndarray) and isinstance(list2, np.ndarray):
        return equals_list(list1.tolist(), list2.tolist())
    elif not isinstance(list1, list) and not isinstance(list2, list) and not isinstance(list1, np.ndarray) and not isinstance(list2, np.ndarray):
        return list1 == list2
    else:
        return False