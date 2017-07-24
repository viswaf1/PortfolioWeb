import datetime, sys, os, multiprocessing, time
import threading
from userPortfolio.models import UserTransactionsModel, UserPortfolioModel, AllStocksModel, SNP500Model, USDForexModel
import csv, hashlib
import pandas, urllib2, csv, random, datetime, string, subprocess
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
    def __init__(self, date='', type='snp500', num_days=500):
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

    def build_classifier(self):
        pass

    def async_callback_svm_features(self, result):
        if result == '':
            return
        stock_name, data = result
        #print stock_name
        if data.shape[0] < self.num_days:
            print "Error in stock data: "+stock_name
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
            print "Error in stock data: "+stock_name
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


    def pick_stocks(self, pick_date = None, stockNames=[]):

        self.all_results = []
        test_period = 1

        #train for number of days and test for test_period
        offset = 50
        slice = self.look_ahead+test_period+self.num_days+offset

        start_ind = slice
        end_ind = 1


        if len(stockNames) == 0:
            if self.type == 'allstocks':
                stocks = AllStocksModel.objects.all()
            elif self.type == 'snp500':
                stocks = SNP500Model.objects.all()
            elif self.type == 'forex':
                stocks = USDForexModel.objects.all()
            stockNames = []
            for eachStock in stocks:
                stockNames.append(eachStock.stockName)


        #append close_snp to the stock data
        all_inputs = []
        backend.StockData.Instance().append_stock_column('^GSPC', 'close', '_index')
        period_stock_data =[]
        self.period_features = {}
        for stockName in stockNames:
            input = []
            try:
                stock_data = backend.StockData.Instance().get_historical_stock_data(stockName)
            except Exception as err:
                print "Error getting data for " + stockName + " " + str(err)
                continue
            if len(stock_data) < 1:
                continue
            if pick_date:
                try:
                    date_ind = stock_data.index.get_loc(pick_date)
                except:
                    print("EnsambleClassifier:pick_stock: date not found for stock "+stockName)
                    continue
                stock_data = stock_data[:date_ind]
            if len(stock_data) > start_ind:
                stock_slice = stock_data[-start_ind:-end_ind]
                if stock_slice.iloc[-1].close > self.min_price:
                    period_stock_data.append((stockName, stock_slice, self.num_days+self.look_ahead, self.look_ahead, offset, True))
        pool = multiprocessing.Pool(16)
        for eachStockData in period_stock_data:
            # ret = portmlp.get_feature_label_for_stocks_raw(eachStockData)
            # ret = get_feature_label_for_stocks(eachStockData)
            # import pdb; pdb.set_trace()
            pool.apply_async(get_feature_label_for_stocks, args=(eachStockData,), callback=self.async_callback_svm_features)
            # pool.apply_async(get_feature_label_for_stocks_raw, args=(eachStockData,), callback=self.async_callback_mlp_features)
        pool.close()
        pool.join()
        period_train_test_data = []
        pf_keys = self.period_features.keys()
        for eachKey in pf_keys:
            eachFeature = self.period_features[eachKey]
            fkeys = eachFeature.keys()
            if len(fkeys) < 1:
                print("Error only one feature found")
                continue
            combined = {}
            for fk in fkeys:
                features = eachFeature[fk]['features']
                labels = eachFeature[fk]['labels']

                train_data = features[:self.num_days]
                train_labels = labels[:self.num_days]
                test_data = [features[-1]]
                test_labels = [1.0]
                if(fk == 'svm'):
                    combined[fk] = (train_data, train_labels, test_data, test_labels, self.C, self.gamma, self.epsilon, eachFeature[fk]['stockName'])

                    negative_train_labels = eachFeature[fk]['negative_labels'][:self.num_days]
                    combined['svm_negative'] = (train_data, negative_train_labels, test_data, test_labels, self.C, self.gamma, self.epsilon, eachFeature[fk]['stockName'])
                    negative_train_labels2 = eachFeature[fk]['negative_labels2'][:self.num_days]
                    combined['svm_negative2'] = (train_data, negative_train_labels2, test_data, test_labels, self.C, self.gamma, self.epsilon, eachFeature[fk]['stockName'])

                else:
                    combined[fk] = (train_data, train_labels, test_data, test_labels, 2, 500, eachFeature[fk]['stockName'])
            period_train_test_data.append(combined)

        self.combined_result = []
        for eachTrainTest in period_train_test_data:
            svm_ret = portsvm.run_libSVMGPU(eachTrainTest['svm'])
            svm_negative_ret = portsvm.run_libSVMGPU(eachTrainTest['svm_negative'])
            svm_negative_ret2 = portsvm.run_libSVMGPU(eachTrainTest['svm_negative2'])
            # mlp_ret = run_MLP(eachTrainTest['mlp'])
            test_labels = eachTrainTest['svm'][3]
            poss = []
            svm_labels = svm_ret['prediction']
            # mlp_labels = mlp_ret['prediction']
            for i in range(len(test_labels)):
                temp_dic = {'svm_ret':svm_ret, 'svm_negative_ret':svm_negative_ret, 'svm_negative_ret2':svm_negative_ret2}
                # temp_dic = {'mlp_ret':mlp_ret}
                temp_dic['actual'] = test_labels[i]
                self.combined_result.append(temp_dic)
        #picked_stocks = [x['svm_ret']['stockName'] for x in self.combined_result if x['svm_ret']['prediction'][0] > 0.1 and x['mlp_ret']['prediction'][0] > 0.1 and x['mlp_ret']['training_accuracy'] > 0.59 and x['mlp_ret']['training_accuracy'] < 0.69 and x['svm_ret']['training_accuracy'] < 0.60 and x['svm_ret']['training_accuracy'] > 0.30]
        picked_stocks = [x['svm_ret']['stockName'] for x in self.combined_result if x['svm_ret']['prediction'][0] > 0.1 and x['svm_negative_ret']['prediction'][0] > 0.1 and x['svm_negative_ret2']['prediction'][0] > 0.1]
        # picked_stocks = [x['mlp_ret']['stockName'] for x in self.combined_result if x['mlp_ret']['prediction'][0] > 0.1]# and x['mlp_ret']['training_accuracy'] > 0.59 and x['mlp_ret']['training_accuracy'] < 0.69]
        print picked_stocks
        return self.rank_picks(picked_stocks, pick_date)

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


    def future_test(self, test_period=100, num_periods=5, moveback = False, period_start = 0, stockNames = []):
        self.svm_result = []

        self.successes_dic = {}
        self.fails_dic = {}
        self.all_results = []
        self.percent_covered = []
        self.combined_result = []
        for period in range(period_start, num_periods):
            #train for number of days and test for test_period
            offset = 50
            slice = (2*self.look_ahead)+test_period+self.num_days+offset

            if moveback:
                start_ind = ((period+1)*moveback)+slice
                end_ind = ((period)*moveback)+1
            else:
                start_ind = (period+1)*slice
                end_ind = (period)*slice+1


            if len(stockNames) == 0:
                if self.type == 'allstocks':
                    stocks = AllStocksModel.objects.all()
                elif self.type == 'snp500':
                    stocks = SNP500Model.objects.all()
                stockNames = []
                for eachStock in stocks:
                    stockNames.append(eachStock.stockName)


            #append close_snp to the stock data
            all_inputs = []
            backend.StockData.Instance().append_stock_column('^GSPC', 'close', '_index')
            period_stock_data =[]
            self.period_features = {}
            for stockName in stockNames:
                input = []
                try:
                    stock_data = backend.StockData.Instance().get_historical_stock_data(stockName)
                except Exception as err:
                    print "Error getting data for " + stockName + " " + str(err)
                    continue
                if len(stock_data) < 1:
                    continue
                if len(stock_data) > start_ind:
                    stock_slice = stock_data[-start_ind:-end_ind]
                    if stock_slice.iloc[-1].close > self.min_price:
                        period_stock_data.append((stockName, stock_slice, test_period+self.num_days+self.look_ahead, self.look_ahead, offset, False))

            pool = multiprocessing.Pool(16)
            for eachStockData in period_stock_data:
                pool.apply_async(portsvm.get_feature_label_for_stocks, args=(eachStockData,), callback=self.async_callback_svm_features)
                pool.apply_async(portmlp.get_feature_label_for_stocks_raw, args=(eachStockData,), callback=self.async_callback_mlp_features)
            pool.close()
            pool.join()

            period_train_test_data = []
            pf_keys = self.period_features.keys()
            for eachKey in pf_keys:
                eachFeature = self.period_features[eachKey]
                fkeys = eachFeature.keys()
                if len(fkeys) < 2:
                    print("Error only one feature found")
                    continue
                combined = {}
                for fk in fkeys:
                    features = eachFeature[fk]['features']
                    labels = eachFeature[fk]['labels']

                    train_data = features[:self.num_days]
                    train_labels = labels[:self.num_days]
                    test_data = features[self.num_days+self.look_ahead:self.num_days+self.look_ahead+test_period]
                    test_labels = labels[self.num_days+self.look_ahead:self.num_days+self.look_ahead+test_period]
                    if(fk == 'svm'):
                        combined[fk] = (train_data, train_labels, test_data, test_labels, self.C, self.gamma, self.epsilon, eachFeature[fk]['stockName'])
                    else:
                        combined[fk] = (train_data, train_labels, test_data, test_labels, 2, 500, eachFeature[fk]['stockName'])
                period_train_test_data.append(combined)


            for eachTrainTest in period_train_test_data:
                svm_ret = portsvm.run_libSVMGPU(eachTrainTest['svm'])
                mlp_ret = portmlp.run_MLP(eachTrainTest['mlp'])
                test_labels = eachTrainTest['svm'][3]
                poss = []
                svm_labels = svm_ret['prediction']
                mlp_labels = mlp_ret['prediction']

                success = 0
                fails = 0
                for i in range(len(test_labels)):
                    temp_dic = {'svm_ret':svm_ret, 'mlp_ret':mlp_ret}
                    temp_dic['actual'] = test_labels[i]
                    self.combined_result.append(temp_dic)

        return self.svm_result

    def svm_async_callback(self, result):
        # self.svm_result.append(result)
        stockName = result['stockName']
        if result['accuracy'] >= 0:
            if result['accuracy'] == 0:
                x = -1
            if result['accuracy'] > 0:
                x = 1

            if stockName in self.all_results:
                self.all_results[stockName] = self.all_results[stockName] + x
            else:
                self.all_results[stockName] = x


        if result['positive_accuracy'] >= 0:
            self.svm_result.append(result['positive_accuracy'])

            if result['positive_accuracy'] > 0:
                if stockName in self.successes_dic:
                    self.successes_dic[stockName] = self.successes_dic[stockName]+1
                else:
                    self.successes_dic[stockName] = 1
            else:
                if stockName in self.fails_dic:
                    self.fails_dic[stockName] = self.fails_dic[stockName]+1
                else:
                    self.fails_dic[stockName] = 1

        if result['percent_covered'] >= 0:
            self.percent_covered.append(result['percent_covered'])


    def cointoss_async_callback(self, result):
        self.cointoss_results.append(result)


def get_feature_label_for_stocks((stock_name, data, num_days, look_ahead, offset, blind)):
    #import pdb; pdb.set_trace()
    # data = data[:-200]
    # snp_data = snp_data[:-200]
    # nasdaq_data = nasdaq_data[:-200]
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

        # index2_data = data_frame['volume_index2'].values
        # index2_disparity5 = index2_data/talib.MA(index2_data, timeperiod=5)
        # index2_disparity5 = index2_disparity5/abs(np.nanmean(index2_disparity5))
        # index2_disparity5 = np.clip(index2_disparity5, -1, 1)

        # index_disparity5 = index_disparity5/np.nanmax(np.abs(index_disparity5))
        # index_mom = index_mom/np.nanmax(np.abs(index_mom))
        # index_macd_hist = index_macd_hist/np.nanmax(np.abs(index_macd_hist))

        # mom10 = mom10/np.nanmax(np.abs(mom10))
        # mom3 = mom3/np.nanmax(np.abs(mom3))
        # willr10 = willr10/np.nanmax(np.abs(willr10))
        # rsi16 = rsi16/np.nanmax(np.abs(rsi16))
        # cci12 = cci12/np.nanmax(np.abs(cci12))
        # rocr3 = rocr3/np.nanmax(np.abs(rocr3))
        # macdhist = macdhist/np.nanmax(np.abs(macdhist))
        # natr = natr/np.nanmax(np.abs(natr))
        # adx = adx/np.nanmax(np.abs(adx))
        # #nasdaq_mom1 = nasdaq_mom1/np.nanmax(np.abs(nasdaq_mom1))
        #
        # percentk = percentk/np.nanmax(np.abs(percentk))
        # percentd = percentd/np.nanmax(np.abs(percentd))
        # slowpercentd = slowpercentd/np.nanmax(np.abs(slowpercentd))
        # ado = ado/np.nanmax(np.abs(ado))
        # disparity5 = disparity5/np.nanmax(np.abs(disparity5))
        # disparity10 = disparity10/np.nanmax(np.abs(disparity10))
        # ultimate = ultimate/np.nanmax(np.abs(ultimate))
        # beta = beta/np.nanmax(np.abs(beta))
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
        #feature_matrix = np.column_stack((mom10, mom3, rsi16, cci12, macdhist, ado, willr10, disparity5, disparity10, beta, index_macd_hist, labels))
        #*feature_matrix = np.column_stack((mom10, cci12, percentd, beta, disparity10, index_disparity5, ultimate, adx, labels))
        feature_matrix = np.column_stack((mom10, cci12, percentd, beta, disparity10, index_disparity5, ultimate, adx, labels, neg_labels, neg_labels2))
        #feature_matrix = np.column_stack((mom10, mom3, rsi16, cci12, macdhist, percentk, percentd, ado,
         #willr10, disparity5, disparity10, index_mom,
        # shift(disparity10, 1, cval=np.NaN), shift(disparity10, 2, cval=np.NaN), shift(disparity10, 3, cval=np.NaN),
        #  labels))

        #feature_matrix = np.column_stack((mom10, mom3, rsi16, cci12, macdhist, percentk, percentd, ado, willr10, disparity5, disparity10, labels))
        #print("--- %s seconds ---" % (time.time() - start_time))
        if not blind:
            feature_matrix = feature_matrix[offset:-(look_ahead)]
        else:
            feature_matrix = feature_matrix[offset:]
    # except Exception as e:
    #     print ("portsvm:get_feature_label_for_stocks :" + str(e))
    #     feature_matrix = np.array([])
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
