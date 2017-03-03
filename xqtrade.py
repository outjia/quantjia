# coding=utf-8

import easytrader as et
from quantjia import predict_today

stocks = {'601866':12.1}


def adjust_position(pcode, stocks):
    user = et.use('xq')
    user.prepare(user='kaoyanzone@hotmail.com', password='801010', portfolio_code=pcode)

    for s in user.position:
        if s['stock_code'] not in stocks.keys():
            user.sell(s['stock_code'], price=s['last_price'] - 0.01, amount=s['enable_amount'])
        else:
            del stocks[s['stock_code']]
    vol = (user.balance['enable_balance']-100)/len(stocks)

    for s in stocks:
        user.buy(s, stocks[s] + 0.01, volume=vol)


def __main__():
    stocks = predict_today()
    adjust_position('ZH181903',stocks[:10])