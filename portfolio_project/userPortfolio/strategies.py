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
import userPortfolio.backend as backend

def apply_simplestrategy(date='', type='allstocks', num=10):
    smpl = SimpleStrategy(date, type, num)

class SimpleStrategy:
    def __init__(self, date='', type='allstocks', num=10):
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
        print datetime.datetime.now().time()
        for eachStock in self.stocks:
            input.append((date, eachStock.stockName, type))
        pool = multiprocessing.Pool(10)
        result = pool.map(run_simplestrategy, input)
        print "Finished processing"
        selectedStocks = pandas.DataFrame(result, columns=('date', 'type', 'stockName', 'score'))
        selectedStocks = selectedStocks.sort(['score'], ascending=[True])
        print selectedStocks.tail(num)
        print datetime.datetime.now().time()

def run_simplestrategy( (date, stockName, type)):
    score = 0
    output = (date, type, stockName, -1)
    try:
        stock_data = backend.get_historical_stock_data(stockName)
    except Exception as err:
        print "Error getting data for " + stockName + " " + str(err)
        return output
    #get the last 100 datapoints till the date
    if not date in stock_data.index:
        date_time = datetime.datetime.combine(date, datetime.time.min)
        date = (date_time - datetime.timedelta(days=1)).date()
        if not date in stock_data.index:
            date_time = datetime.datetime.combine(date, datetime.time.min)
            date = (date_time - datetime.timedelta(days=1)).date()
            if not date in stock_data.index:
                return output

    data = stock_data.loc[:date].tail(100)

    try:
        macd = MACD(data, fastperiod=12, slowperiod=26, signalperiod=9)
        macd_val = macd.loc[date].macd
        macd_signal = macd.loc[date].macdsignal
        macd_slope = macd_val - (macd.iloc[macd.index.get_loc(date)-2].macd) / 2

        if (macd_val > macd_signal) and macd_slope > 0:
            #score = score + 1#(macd_val - macd_signal)
            #score = score + (macd_val-macd_signal)
            for p in range(1, 5):
                loc = macd.index.get_loc(date)
                prev_macd_val = macd.iloc[loc-p].macd
                prev_macd_signal = macd.iloc[loc-p].macdsignal
                if prev_macd_signal > prev_macd_val:
                    score = score + 1
                    score = score + (macd_val-macd_signal)
                    break

        cci = CCI(data, timeperiod=14)
        cci_val = cci.loc[date]
        cci_slope = cci_val-cci.iloc[cci.index.get_loc(date)-2]
        # print date
        # print cci_val
        # print cci.irow(cci.index.get_loc(date)-2)
        if cci_val >= 0:# and cci_slope > 0:
            #score = score + 1
            #check if zero cross over is within the last 5 days
            for p in range(1, 5):
                loc = cci.index.get_loc(date)
                prev_cci = cci.iloc[loc-p]
                if prev_cci < 0:
                    score = score + 1
                    break
        if cci_val > 100 and cci_slope > 0 and cci_val < 200:
             score = score + 1

        adx = ADX(data, timeperiod=14)
        plusdm = PLUS_DM(data, timeperiod=14)
        minusdm = MINUS_DM(data, timeperiod=14)

        if adx.loc[date] > 25:
            if plusdm.loc[date] > minusdm.loc[date]:
                score = score + 1

        series = (date, type, stockName, score)
        #selectedStocks.loc[len(selectedStocks.index)] = series
        print stockName + " : " + str(score)
        return series
    except:
        return output






#
