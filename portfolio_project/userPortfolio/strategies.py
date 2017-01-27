import datetime, sys, os, multiprocessing
from userPortfolio.models import UserTransactionsModel, UserPortfolioModel, AllStocksModel, SNP500Model
from django.db import IntegrityError
from django.conf import settings
import csv, hashlib
from pandas_datareader import data
import pandas, urllib2, csv
from pytz import timezone
from dateutil.relativedelta import relativedelta
from math import pi
from talib.abstract import *
import talib
import talib
import userPortfolio.backend as backend

def apply_simplestrategy(date='', type='allstocks', num=100):
    smpl = SimpleStrategy(date, type, num)

class SimpleStrategy:
    def __init__(self, date='', type='allstocks', num=100):
        self.date = date
        self.type = type
        self.num = num
        self.trend = 19
        self.ema_span = 13
        self.results = []

    def async_callback(self, result):
        self.results.append(result)

    def run(self):
        date = self.date
        type = self.type
        num = self.num
        if date == '':
            currentTimeUtc = datetime.datetime.now(timezone('UTC'))
            nycTime = currentTimeUtc.astimezone(timezone('US/Eastern'))
            self.date = nycTime.date()
        if type == 'allstocks':
            self.stocks = AllStocksModel.objects.all()
        elif type == 'snp500':
            self.stocks = SNP500Model.objects.all()

        #self.selectedStocks = pandas.DataFrame(columns=('date', 'type', 'stockName', 'score'))

        input = []
        #print(datetime.datetime.now().time())
        for eachStock in self.stocks:
            stockName = eachStock.stockName
            if (eachStock.stockName == 'EMC') or (eachStock.stockName == 'TYC'):
                continue
            try:
                stock_data = backend.StockData.Instance().get_historical_stock_data(stockName)
            except Exception as err:
                print "Error getting data for " + stockName + " " + str(err)
                continue
            if stock_data.iloc[-1].close >= 50:
                input.append((date, eachStock.stockName, type, self.trend, self.ema_span, stock_data))
        pool = multiprocessing.Pool(50)
        for eachInput in input:
            pool.apply_async(runPQRules, args=(eachInput,), callback=self.async_callback)
        pool.close()
        pool.join()
        #result = pool.map(run_simplestrategy, input)
        #result = pool.map(runPQRules, input)

        #print("Finished processing")
        selectedStocks = pandas.DataFrame(self.results, columns=('date', 'type', 'stockName', 'score'))
        selectedStocks = selectedStocks.sort(['score'], ascending=[True])
        #print(datetime.datetime.now().time())
        pool.terminate()

        return selectedStocks[selectedStocks.score > 0]


def runPQRules( (date, stockName, type, trend, ema_span, stock_data)):
    score = 0
    output = (date, type, stockName, -1)
    # try:
    #     stock_data = backend.StockData.Instance().get_historical_stock_data(stockName)
    # except Exception as err:
    #     print "Error getting data for " + stockName + " " + str(err)
    #     return output
    #get the last 100 datapoints till the date
    if not date in stock_data.index:
        return output
        date_time = datetime.datetime.combine(date, datetime.time.min)
        date = (date_time - datetime.timedelta(days=1)).date()
        if not date in stock_data.index:
            date_time = datetime.datetime.combine(date, datetime.time.min)
            date = (date_time - datetime.timedelta(days=1)).date()
            if not date in stock_data.index:
                return output

    data = stock_data.loc[:date].tail(50)

    try:
        close = data['close'].values
        #price_ema_data = pandas.ewma(data['close'], span=ema_span)
        #price_ema_data = data['close'].ewm(ignore_na=False,span=ema_span,min_periods=0,adjust=True).mean()
        price_ema_data = talib.EMA(close, timeperiod=ema_span)
        rsi_data = talib.RSI(close, timeperiod=14)
        loc = data.index.get_loc(date)
        #rsi_ema_data = pandas.ewma(rsi_data, span=5)
        #rsi_ema_data = rsi_data.ewm(ignore_na=False,span=5,min_periods=0,adjust=True).mean()
        rsi_ema_data = talib.EMA(rsi_data, timeperiod=5)
        rsi = rsi_data[loc]
        rsi_ema = rsi_ema_data[loc]


        price_cur = price_ema_data[loc]
        price_prev = price_ema_data[loc-50]
        if check_trend(price_ema_data, trend, loc):
            if rsi > rsi_ema and rsi_ema > 40 and rsi_ema < 60:
                score = score + 10

        series = (date, type, stockName, score)
        # print stockName + " : " + str(score)
        return series
    except:
        return output

def check_trend(data, span, ind):
    for i in range(span):
        newInd = ind-1
        val = data[ind]
        old_val = data[newInd]
        ind = newInd
        if old_val > val:
            return False
    return True



#-------------------------------------------------------------------------------
