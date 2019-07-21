import datetime, sys, os, multiprocessing, time, pickle
import threading
from userPortfolio.models import UserTransactionsModel, UserPortfolioModel, AllStocksModel, SNP500Model, USDForexModel
import csv, hashlib
import pandas, urllib.request, urllib.error, urllib.parse, csv, random, datetime, string, subprocess
from pytz import timezone
from dateutil.relativedelta import relativedelta
from math import pi
import userPortfolio.backend as backend
import numpy as np
from sklearn import svm
from scipy.ndimage.interpolation import shift
import userPortfolio.portsvm as portsvm
import userPortfolio.portmlp as portmlp
from operator import itemgetter
import talib

import theano
import theano.tensor as T
import lasagne


class EnsambleClassifier:
    def __init__(self, date='', type='snp500', num_days=200):
        self.date = date
        self.type = type
        self.look_ahead = 10
        self.min_price = 50
        self.gamma = 1.3
        self.C = 500
        self.epsilon = 0.001
        self.num_days = num_days
        self.results = []
        self.labels = []
        self.period_features_lock = threading.Lock()
        self.svm_predictions_dictionary = {}
        self.svm_data = "svm_data/"

    def build_classifier(self):
        pass

    def async_callback_svm_features(self, result):
        if result == '':
            return
        stock_name, data = result
        #print stock_name
        if data.shape[0] < self.num_days:
            print("Error in stock data: "+stock_name)
            return

        features = []
        labels = []
        negative_labels = []
        negative_labels2 = []
        for i in range(len(data)):

            if np.isnan(data[i][-2]):
                negative_labels2.append(np.nan)
            elif data[i][-2] < 0:
                negative_labels2.append(-1)
            elif data[i][-2] > 0:
                negative_labels2.append(1)

            if np.isnan(data[i][-3]):
                labels.append(np.nan)
            elif data[i][-3] < 0:
                labels.append(-1)
            elif data[i][-3] > 0:
                labels.append(1)

            if np.isnan(data[i][-1]):
                negative_labels.append(np.nan)
            elif data[i][-1] < 0:
                negative_labels.append(-1)
            elif data[i][-1] > 0:
                negative_labels.append(1)

            features.append(data[i][:-3].tolist())
        self.period_features_lock.acquire()
        if stock_name in self.period_features:
            self.period_features[stock_name]['svm'] = {'stockName': stock_name, 'features':features, 'labels':labels, 'negative_labels':negative_labels, 'negative_labels2':negative_labels2}
        else:
            self.period_features[stock_name] = {'svm' :{'stockName': stock_name, 'features':features, 'labels':labels, 'negative_labels':negative_labels, 'negative_labels2':negative_labels2}}
        self.period_features_lock.release()


    def async_callback_mlp_features(self, result):
        if result == '':
            return
        stock_name, data = result
        #print stock_name
        if data.shape[0] < self.num_days:
            print("Error in stock data: "+stock_name)
            return

        features = []
        labels = []
        for i in range(len(data)):
            if np.isnan(data[i][-1]):
                labels.append(np.nan)
            elif data[i][-1] < 0:
                labels.append(0)
            elif data[i][-1] > 0:
                labels.append(1)

            features.append(data[i][:-1].tolist())
        self.period_features_lock.acquire()
        if stock_name in self.period_features:
            self.period_features[stock_name]['mlp'] = {'stockName': stock_name, 'features':features, 'labels':labels}
        else:
            self.period_features[stock_name] = {'mlp' :{'stockName': stock_name, 'features':features, 'labels':labels}}

        self.period_features_lock.release()



    def rank_picks(self, picked_stocks, date):
        stocks_scores = []
        for eachStock in picked_stocks:
            stock_data = backend.StockData.Instance().get_historical_stock_data(eachStock)
            stock_data = stock_data.ix[:date]
            close = stock_data['close'].values
            macd, macdsignal, macdhist = talib.MACD(close[-100:])
            stocks_scores.append({'stock':eachStock, 'score':macdhist[-1]})
        sorted_stocks = sorted(stocks_scores, key=itemgetter('score'))
        return [x['stock'] for x in sorted_stocks]



    def run_predicion_for_n_days(self, days, reset = False):

        offset = 50

        if self.type == 'allstocks':
            stocks = AllStocksModel.objects.all()
        elif self.type == 'snp500':
            stocks = SNP500Model.objects.all()
        elif self.type == 'forex':
            stocks = USDForexModel.objects.all()
        stockNames = []
        for eachStock in stocks:
            stockNames.append(eachStock.stockName)

        backend.StockData.Instance().append_stock_column('^GSPC', 'close', '_index')

        for stockName in stockNames:
            try:
                stock_data = backend.StockData.Instance().get_historical_stock_data(stockName)
            except Exception as err:
                print("Error getting data for " + stockName + " " + str(err))
                continue
            if len(stock_data) < 1:
                continue


    def load_svm_predictions_dictionary(self):
        try:
            self.svm_predictions_dictionary = pickle.load(open(self.svm_predictions_file, 'rb'))
        except Exception as ex:
            print((str(ex)))
            self.svm_predictions_dictionary = {}


    def svm_async_callback(self, result):
        self.svm_result.append(result)

    def cointoss_async_callback(self, result):
        self.cointoss_results.append(result)


def get_feature_label_for_stocks(xxx_todo_changeme):
    #import pdb; pdb.set_trace()
    # data = data[:-200]
    # snp_data = snp_data[:-200]
    # nasdaq_data = nasdaq_data[:-200]
    (stock_name, data, num_days, look_ahead, offset, blind) = xxx_todo_changeme
    start_time = time.time()
    # offset = 40
    slice_len = num_days+look_ahead+offset
    data = data.tail(slice_len)
    #nasdaq_data = nasdaq_data.tail(slice_len)
    #data_frame = pandas.merge(data, snp_data.close.to_frame(), suffixes=('', '_snp'), left_index=True, right_index=True)
    #data_frame = pandas.merge(data, nasdaq_data.close.to_frame(), suffixes=('', '_nasdaq'), left_index=True, right_index=True)
    data_frame = data
    index_data = data_frame['close_index'].values
    data = data_frame['close'].values
    high = data_frame['high'].values
    low = data_frame['low'].values
    volume = data_frame['volume'].values

    if 1:
        #Append required technical indicators to the data
        mom10 = talib.MOM(data, timeperiod=10)
        mom3 = talib.MOM(data, timeperiod=3)
        willr10 = talib.WILLR(high, low, data)
        rsi16 = talib.RSI(data)
        cci12 = talib.CCI(high, low, data)
        rocr3 = talib.ROCR(data)
        macd, macdsignal, macdhist = talib.MACD(data)

        percentk, percentd = talib.STOCHF(high, low, data)
        slowpercentk, slowpercentd = talib.STOCH(high, low, data)
        #ado = talib.ADOSC(high, low, data, volume)
        natr = talib.NATR(high, low, data)
        ultimate = talib.ULTOSC(high, low, data)

        ma5 = talib.MA(data, timeperiod=5)
        ma10 = talib.MA(data, timeperiod=10)
        disparity5 = data/ma5
        disparity10 = data/ma10
        beta = talib.BETA(high, low)
        adx = talib.ADX(high, low, data)

        index_disparity5 = index_data/talib.MA(index_data, timeperiod=5)

        index_disparity5 = index_disparity5/abs(np.nanmean(index_disparity5))

        mom10 = mom10/abs(np.nanmean(mom10))
        mom3 = mom3/abs(np.nanmean(mom3))
        willr10 = willr10/abs(np.nanmean(willr10))
        rsi16 = rsi16/abs(np.nanmean(rsi16))
        cci12 = cci12/abs(np.nanmean(cci12))
        rocr3 = rocr3/abs(np.nanmean(rocr3))
        macdhist = macdhist/abs(np.nanmean(macdhist))
        natr = natr/abs(np.nanmean(natr))
        adx = adx/abs(np.nanmean(adx))
        #nasdaq_mom1 = nasdaq_mom1/abs(np.nanmean(nasdaq_mom1))

        percentk = percentk/abs(np.nanmean(percentk))
        percentd = percentd/abs(np.nanmean(percentd))
        slowpercentd = slowpercentd/abs(np.nanmean(slowpercentd))
        #ado = ado/abs(np.nanmean(ado))
        disparity5 = disparity5/abs(np.nanmean(disparity5))
        disparity10 = disparity10/abs(np.nanmean(disparity10))
        ultimate = ultimate/abs(np.nanmean(ultimate))
        beta = beta/abs(np.nanmean(beta))

        mom10 = np.clip(mom10, -1, 1)
        cci12 = np.clip(cci12, -1, 1)
        percentd = np.clip(percentd, -1, 1)
        disparity10 = np.clip(disparity10, -1, 1)
        index_disparity5 = np.clip(index_disparity5, -1, 1)
        ultimate = np.clip(ultimate, -1, 1)
        adx = np.clip(adx, -1, 1)

        mom3 = np.clip(mom3, -1, 1)
        rsi16 = np.clip(rsi16, -1, 1)
        macdhist = np.clip(macdhist, -1, 1)
        willr10 = np.clip(willr10, -1, 1)
        disparity5 = np.clip(disparity5, -1, 1)
        percentk = np.clip(percentk, -1, 1)

        beta = np.clip(beta, -1, 1)

        labels = get_label(data, look_ahead)
        neg_labels = get_negative_label(data, look_ahead)
        neg_labels2 = get_negative_label2(data, look_ahead)

        feature_matrix = np.column_stack((mom10, cci12, percentd, beta, disparity10, index_disparity5, ultimate, adx, labels, neg_labels, neg_labels2))

        if not blind:
            feature_matrix = feature_matrix[offset:-(look_ahead)]
        else:
            feature_matrix = feature_matrix[offset:]

    return (stock_name, feature_matrix)



#one if price increases by 5% in look ahead
def get_label(data, look_ahead):
    #data - numpy array
    labels = np.array([])
    data_len = data.size
    for i in range(data_len):
        if i+look_ahead >= data_len:
            labels = np.append(labels, np.nan)
            continue
        change = data[i+look_ahead] - data[i]
        if change > 0.0 and (change/data[i])*100 >= 5.0:
            found = True
            labels = np.append(labels, 1)
        else:
            labels = np.append(labels, -1)
    return labels


def get_negative_label(data, look_ahead):
    #data - numpy array
    labels = np.array([])
    data_len = data.size
    for i in range(data_len):
        if i+look_ahead >= data_len:
            labels = np.append(labels, np.nan)
            continue
        sma = 0
        found = False
        for s in range(1,look_ahead+1):
            change = data[i+s]-data[i]
            if change < 0:
                found = True
                labels = np.append(labels, -1)
                break

        if not found:
            labels = np.append(labels, 1)
        # change = data[i+look_ahead] - data[i]
        # if change < 0:# and (change/data[i])*100 >= 1.0:
        #     found = True
        #     labels = np.append(labels, -1)
        # else:
        #     labels = np.append(labels, 1)
    return labels

def get_negative_label2(data, look_ahead):
    #data - numpy array
    labels = np.array([])
    data_len = data.size
    for i in range(data_len):
        if i+look_ahead >= data_len:
            labels = np.append(labels, np.nan)
            continue
        sma = 0
        found = False
        change = data[i+look_ahead] - data[i]
        if change < 0:# and (change/data[i])*100 >= 1.0:
            found = True
            labels = np.append(labels, -1)
        else:
            labels = np.append(labels, 1)
    return labels

def get_feature_label_for_stocks_raw(xxx_todo_changeme1):
    #import pdb; pdb.set_trace()
    # data = data[:-200]
    # snp_data = snp_data[:-200]
    # nasdaq_data = nasdaq_data[:-200]
    (stock_name, data, num_days, look_ahead, offset, blind) = xxx_todo_changeme1
    start_time = time.time()
    # offset = 40
    slice_len = num_days+look_ahead+offset
    data = data.tail(slice_len)
    #nasdaq_data = nasdaq_data.tail(slice_len)
    #data_frame = pandas.merge(data, snp_data.close.to_frame(), suffixes=('', '_snp'), left_index=True, right_index=True)
    #data_frame = pandas.merge(data, nasdaq_data.close.to_frame(), suffixes=('', '_nasdaq'), left_index=True, right_index=True)
    data_frame = data
    index_data = data_frame['close_index'].values
    data = data_frame['close'].values
    high = data_frame['high'].values
    low = data_frame['low'].values
    volume = data_frame['volume'].values


    try:
        #Append required technical indicators to the data

        # close = data/np.nanmax(np.abs(data))
        # high = high/np.nanmax(np.abs(high))
        # low = low/np.nanmax(np.abs(low))
        # volume = volume/np.nanmax(np.abs(volume))
        # index_data = index_data/np.nanmax(np.abs(index_data))


        # close = data/abs(np.mean(data))
        # high = high/abs(np.mean(high))
        # low = low/abs(np.mean(low))
        # volume = volume/abs(np.mean(volume))
        # index_data = index_data/abs(np.mean(index_data))


        close = data/abs(np.mean(data))
        close = np.clip(close, -0.99, 0.99)

        high = high/abs(np.mean(high))
        high = np.clip(high, -0.99, 0.99)

        low = low/abs(np.mean(low))
        low = np.clip(low, -0.99, 0.99)

        volume = volume/abs(np.mean(volume))
        volume = np.clip(volume, -0.99, 0.99)

        index_data = index_data/abs(np.mean(index_data))
        index_data = np.clip(index_data, -0.99, 0.99)

        # close = (close*2)-1
        # high = (high*2)-1
        # low = (low*2)-1
        # volume = (volume*2)-1
        # index_data = (index_data*2)-1


        labels = get_label(data, look_ahead)


        #feature_matrix = np.column_stack((mom10, mom3, rsi16, cci12, macdhist, ado, willr10, disparity5, disparity10, beta, index_macd_hist, labels))
        feature_matrix = np.column_stack((close, high, low, volume,
        shift(close, 1, cval=np.NaN), shift(high, 1, cval=np.NaN), shift(low, 1, cval=np.NaN),shift(volume, 1, cval=np.NaN),
        shift(close, 2, cval=np.NaN), shift(high, 2, cval=np.NaN), shift(low, 2, cval=np.NaN),shift(volume, 2, cval=np.NaN),
        shift(close, 3, cval=np.NaN), shift(high, 3, cval=np.NaN), shift(low, 3, cval=np.NaN),shift(volume, 3, cval=np.NaN),
        shift(close, 4, cval=np.NaN), shift(high, 4, cval=np.NaN), shift(low, 4, cval=np.NaN),shift(volume, 4, cval=np.NaN),
        # shift(close, 5, cval=np.NaN), shift(high, 5, cval=np.NaN), shift(low, 5, cval=np.NaN),shift(volume, 5, cval=np.NaN),
        # shift(close, 6, cval=np.NaN), shift(high, 6, cval=np.NaN), shift(low, 6, cval=np.NaN),shift(volume, 6, cval=np.NaN),
        # shift(close, 7, cval=np.NaN), shift(high, 7, cval=np.NaN), shift(low, 7, cval=np.NaN),shift(volume, 7, cval=np.NaN),
        # shift(close, 8, cval=np.NaN), shift(high, 8, cval=np.NaN), shift(low, 8, cval=np.NaN),shift(volume, 8, cval=np.NaN),
        # shift(close, 9, cval=np.NaN), shift(high, 9, cval=np.NaN), shift(low, 9, cval=np.NaN),shift(volume, 9, cval=np.NaN),
        # shift(close, 10, cval=np.NaN), shift(high, 10, cval=np.NaN), shift(low, 10, cval=np.NaN),shift(volume, 10, cval=np.NaN),
        index_data, shift(index_data, 1, cval=np.NaN), shift(index_data, 2, cval=np.NaN), shift(index_data, 3, cval=np.NaN),
        labels))
        # feature_matrix = np.column_stack((mom10, mom3, rsi16, cci12, macdhist, percentk, percentd, ado,
        # willr10, disparity5, disparity10, index_mom,
        # shift(disparity10, 1, cval=np.NaN), shift(disparity10, 2, cval=np.NaN), shift(disparity10, 3, cval=np.NaN),
        #  labels))
        #feature_matrix = np.column_stack((mom10, mom3, rsi16, cci12, macdhist, percentk, percentd, ado, willr10, disparity5, disparity10, labels))
        #print("--- %s seconds ---" % (time.time() - start_time))
        if not blind:
            feature_matrix = feature_matrix[offset:-(look_ahead)]
        else:
            feature_matrix = feature_matrix[offset:]
    except Exception as e:
        print(("portmlp: get_feature_label_for_stocks_raw : " + str(e)))
        feature_matrix = np.array([])
    return (stock_name, feature_matrix)
