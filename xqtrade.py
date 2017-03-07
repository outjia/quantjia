# coding=utf-8

import easytrader as et
import pandas as pd
import numpy as np
from quantjia import predict_today_simple
from DataManager import int2str


def adjust_position(pcode, stocks):
    if len(stocks) ==0: return
    user = et.use('xq')
    user.prepare(user='kaoyanzone@hotmail.com', password='801010', portfolio_code=pcode)

    for s in user.position:
        if s['stock_code'] not in stocks.code:
            user.sell(s['stock_code'], price=s['last_price'] - 0.01, amount=s['enable_amount'])
        else:
            stocks = stocks.drop([stocks['code']==s['stock_code']], axis=0)

    vol = (user.balance[0]['enable_balance']-100)/len(stocks)

    stocks = np.array(stocks)
    for s in stocks:
        user.buy(int2str(s[0]), s[1] + 0.01, volume=vol)


def __main__():
    stocks = predict_today_simple('M1_T5_B256_C3_E100_S2000')
    adjust_position('ZH1036194',stocks[stocks.proba>0.6][:10])
    stocks = predict_today_simple('M1_T10_B256_C4_E100_S100')
    adjust_position('ZH1037189', stocks[stocks.proba>0.5][:10])

if __name__ == '__main__':
    __main__()