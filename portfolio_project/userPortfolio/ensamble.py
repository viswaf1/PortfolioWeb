import datetime, sys, os, multiprocessing, time
import threading
from userPortfolio.models import UserTransactionsModel, UserPortfolioModel, AllStocksModel, SNP500Model
import csv, hashlib
import pandas, urllib2, csv, random, datetime, string, subprocess
from pytz import timezone
from dateutil.relativedelta import relativedelta
from math import pi
from talib.abstract import *
import talib
import userPortfolio.backend as backend
import numpy as np
from sklearn import svm
from scipy.ndimage.interpolation import shift
import userPortfolio.portsvm as portsvm
import userPortfolio.portmlp2 as portmlp2




class EnsambleClassifier:
    def __init__(self, date='', type='snp500', num_days=100):
        self.date = date
        self.type = type
        self.look_ahead = 5
        self.min_price = 200
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
        for i in range(len(data)):

            if data[i][-1] == np.nan:
                continue
            elif data[i][-1] < 0:
                labels.append(-1)
            elif data[i][-1] > 0:
                labels.append(1)

            features.append(data[i][:-1].tolist())
        self.period_features_lock.acquire()
        if stock_name in self.period_features:
            self.period_features[stock_name]['svm'] = {'stockName': stock_name, 'features':features, 'labels':labels}
        else:
            self.period_features[stock_name] = {'svm' :{'stockName': stock_name, 'features':features, 'labels':labels}}
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

            if data[i][-1] == np.nan:
                continue
            elif data[i][-1] < 0:
                labels.append(-1)
            elif data[i][-1] > 0:
                labels.append(1)

            features.append(data[i][:-1].tolist())
        self.period_features_lock.acquire()
        if stock_name in self.period_features:
            self.period_features[stock_name]['mlp'] = {'stockName': stock_name, 'features':features, 'labels':labels}
        else:
            self.period_features[stock_name] = {'mlp' :{'stockName': stock_name, 'features':features, 'labels':labels}}

        self.period_features_lock.release()



    def future_test(self, test_period=100, num_periods=5, moveback = False, period_start = 0, stockNames = []):
        self.svm_result = []

        self.successes_dic = {}
        self.fails_dic = {}
        self.all_results = {}
        self.percent_covered = []
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
                        period_stock_data.append((stockName, stock_slice, test_period+self.num_days+self.look_ahead, self.look_ahead))

            pool = multiprocessing.Pool(16)
            for eachStockData in period_stock_data:
                #return get_feature_label_for_stocks(eachStockData)
                pool.apply_async(portsvm.get_feature_label_for_stocks, args=(eachStockData,), callback=self.async_callback_svm_features)
                pool.apply_async(portmlp2.get_feature_label_for_stocks, args=(eachStockData,), callback=self.async_callback_mlp_features)
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
                        combined[fk] = (all_train, all_train_labels, all_test, all_test_labels, 2, 500, eachFeature['stockName'])
                period_train_test_data.append(combined)
            return period_train_test_data

            #self.svm_result = []
            pool = multiprocessing.Pool(16)
            for eachTrainTest in period_train_test_data:
                pool.apply_async(run_libSVMGPU, args=(eachTrainTest,), callback=self.svm_async_callback)
                #ret = run_libSVMGPU(eachTrainTest)
            pool.close()
            pool.join()


            for eachTrainTest in period_train_test_data:
                svm_ret = portsvm.run_libSVMGPU(eachTrainTest['svm'])
                mlp_ret = portmlp2.run_MLP(eachTrainTest['mlp'])




        #backend.StockData.Instance().flush()

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

#
