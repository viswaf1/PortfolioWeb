import datetime, sys, os, multiprocessing, time, cPickle
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
        self.svm_predictions_file = 'svm_predictions_for_neural_ensamble.txt'

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

    def get_svm_predition_date_stock(self, pick_date,stock_name):
        if  len(self.svm_predictions_dictionary.keys()) < 1:
            self.load_svm_predictions_dictionary()

        date_str = pick_date.strftime('%m%d%Y')
        if (date_str not in self.svm_predictions_dictionary):
            pick_date, predictions = self.run_svm_prediction(pick_date, stock_name)
        elif stock_name not in self.svm_predictions_dictionary[date_str]:
            pick_date, predictions = self.run_svm_prediction(pick_date, stock_name)
        else:
            predictions = self.svm_predictions_dictionary[date_str]

        return predictions

    def run_predicion_for_n_days(self, days):

        self.load_svm_predictions_dictionary()

        stockData = backend.StockData.Instance()
        index_data = stockData.get_historical_stock_data('^GSPC')

        end_ind = len(index_data.index)-1
        start_ind = end_ind-days

        for dayInd in range(start_ind, end_ind+1-self.look_ahead):
            current_date = index_data.index[dayInd].date()
            date_str = current_date.strftime('%m%d%Y')
            if(date_str not in self.svm_predictions_dictionary):
                pick_date, predictions = self.run_svm_prediction(current_date)
                self.svm_predictions_dictionary[date_str] = predictions
                cPickle.dump(self.svm_predictions_dictionary, open(self.svm_predictions_file, 'wb'))
            else:
                print "Date present in dataset"

    def load_svm_predictions_dictionary(self):
        try:
            self.svm_predictions_dictionary = cPickle.load(open(self.svm_predictions_file, 'rb'))
        except Exception as ex:
            print(str(ex))
            self.svm_predictions_dictionary = {}

    def run_svm_prediction(self, pick_date, stock_name=None):
        self.all_results = []
        test_period = 1

        #train for number of days and test for test_period
        offset = 50
        slice = self.look_ahead+test_period+self.num_days+offset

        start_ind = slice
        end_ind = 1

        if not stock_name == None:
            stockNames = [stock_name]
        else:
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

        combined_result = {}
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
                combined_result[svm_ret['stockName']] = temp_dic
        #picked_stocks = [x['svm_ret']['stockName'] for x in self.combined_result if x['svm_ret']['prediction'][0] > 0.1 and x['mlp_ret']['prediction'][0] > 0.1 and x['mlp_ret']['training_accuracy'] > 0.59 and x['mlp_ret']['training_accuracy'] < 0.69 and x['svm_ret']['training_accuracy'] < 0.60 and x['svm_ret']['training_accuracy'] > 0.30]
        #picked_stocks = [x['svm_ret']['stockName'] for x in self.combined_result if x['svm_ret']['prediction'][0] > 0.1 and x['svm_negative_ret']['prediction'][0] > 0.1 and x['svm_negative_ret2']['prediction'][0] > 0.1]
        # picked_stocks = [x['mlp_ret']['stockName'] for x in self.combined_result if x['mlp_ret']['prediction'][0] > 0.1]# and x['mlp_ret']['training_accuracy'] > 0.59 and x['mlp_ret']['training_accuracy'] < 0.69]
        return (pick_date,combined_result);

    def pick_stocks_neural_ensamble(self, pick_date = None, stockNames=[]):

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
        self.period_svm_predictions = {}
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
                    dates = stock_slice.index
                    svm_predictions_stock_date = []
                    skip_stock = False;
                    for each_n_date in dates:
                        each_date = each_n_date.date()
                        svm_pred = self.get_svm_predition_date_stock(each_date, stockName)
                        if stockName not in svm_pred:
                            skip_stock = True
                            print "Skipping svm prediction for stock "+stockName
                            break
                        svm_predictions_stock_date.append([svm_pred[stockName]['svm_ret']['prediction'][0],
                         svm_pred[stockName]['svm_negative_ret']['prediction'][0], svm_pred[stockName]['svm_negative_ret2']['prediction'][0] ])
                    if skip_stock:
                        continue
                    period_stock_data.append((stockName, stock_slice, self.num_days+self.look_ahead, self.look_ahead, offset, True))
                    self.period_svm_predictions[stockName] = svm_predictions_stock_date
        pool = multiprocessing.Pool(16)
        for eachStockData in period_stock_data:
            # ret = portmlp.get_feature_label_for_stocks_raw(eachStockData)
            # ret = get_feature_label_for_stocks(eachStockData)
            # import pdb; pdb.set_trace()
            #pool.apply_async(get_feature_label_for_stocks, args=(eachStockData,), callback=self.async_callback_svm_features)
            pool.apply_async(get_feature_label_for_stocks_raw, args=(eachStockData,), callback=self.async_callback_mlp_features)
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
                for i in range(len(train_data)):
                    train_data[i].extend(self.period_svm_predictions[eachFeature[fk]['stockName']][i])

                train_labels = labels[:self.num_days]
                test_data = [features[-1]].extend(self.period_svm_predictions[eachFeature[fk]['stockName']][-1])
                test_labels = [1.0]

                combined[fk] = (train_data, train_labels, test_data, test_labels, 2, 500, eachFeature[fk]['stockName'])

            period_train_test_data.append(combined)
        import pdb; pdb.set_trace()

        self.combined_result = []
        for eachTrainTest in period_train_test_data:
            # svm_ret = portsvm.run_libSVMGPU(eachTrainTest['svm'])
            # svm_negative_ret = portsvm.run_libSVMGPU(eachTrainTest['svm_negative'])
            # svm_negative_ret2 = portsvm.run_libSVMGPU(eachTrainTest['svm_negative2'])
            mlp_ret = run_MLP(eachTrainTest['mlp'])
            test_labels = eachTrainTest['mlp'][3]
            # poss = []
            # svm_labels = svm_ret['prediction']
            # mlp_labels = mlp_ret['prediction']
            for i in range(len(test_labels)):
                temp_dic = {'mlp_ret':mlp_ret}
                temp_dic['actual'] = test_labels[i]
                import pdb; pdb.set_trace()
                self.combined_result.append(temp_dic)
        print self.combined_result
        # return self.rank_picks(picked_stocks, pick_date)



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
        if change > 0.0 and (change/data[i])*100.0 >= 5.0:
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

def get_feature_label_for_stocks_raw((stock_name, data, num_days, look_ahead, offset, blind)):
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
        print ("portmlp: get_feature_label_for_stocks_raw : " + str(e))
        feature_matrix = np.array([])
    return (stock_name, feature_matrix)

def run_MLP((train, train_labels, test, test_labels, depth, width, stockName)):

    X_train = np.array(train, dtype=np.float32)
    y_train = np.array(train_labels, dtype=np.int32)
    X_test = np.array(test, dtype=np.float32)
    y_test = np.array(test_labels, dtype=np.int32)

    #drop_input=.2
    #drop_hidden=.5

    drop_input=False
    drop_hidden=False
    num_epochs = 30

    input_var = T.matrix('inputs')
    target_var = T.ivector('targets')

    feature_len = (X_train[0].shape)[0]
    network = lasagne.layers.InputLayer(shape=(None, feature_len),
                                        input_var=input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.rectify
    for _ in range(depth):
        network = lasagne.layers.DenseLayer(
                network, width, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, 2, nonlinearity=softmax)


    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.001, momentum=0.9)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()

    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))

    all_train_loss = []
    all_test_loss = []
    all_test_accuracies = []
    final_train_accuracy = 0
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 200, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1


        # # Then we print the results for this epoch:
        # print("Epoch {} of {} took {:.3f}s".format(
        #     epoch + 1, num_epochs, time.time() - start_time))
        # print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        # all_train_loss.append(train_err / train_batches)
        #
        # # After training, we compute and print the test error:
        # test_err = 0
        # test_acc = 0
        # test_batches = 0
        # for batch in iterate_minibatches(X_test, y_test, 2, shuffle=False):
        #     inputs, targets = batch
        #     err, acc = val_fn(inputs, targets)
        #     test_err += err
        #     test_acc += acc
        #     test_batches += 1
        # all_test_accuracies.append(test_acc / test_batches)

    train_loss, final_train_accuracy = val_fn(X_train, y_train)
    test_err, test_acc = val_fn(X_test, y_test)
    test_err = 0+test_err
    test_acc = 0+test_acc
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc * 100))

    result_labels = predict_fn(X_test)

    positive_accuracy = 0
    pos_success = 0
    pos_fails = 0
    suceess = 0
    fails = 0
    for i in range(len(result_labels)):
        if result_labels[i] > 0.1:
            if y_test[i] > 0.1:
                pos_success = pos_success + 1
            else:
                pos_fails = pos_fails + 1


    if pos_success+pos_fails > 0:
        positive_accuracy = (pos_success*100.0)/(pos_success+pos_fails)
    else:
        positive_accuracy = -1

    covered = 0
    missed = 0
    for i in range(len(y_test)):
        if y_test[i] > 0.1:
            if result_labels[i] > 0.1:
                covered = covered + 1
            else:
                missed = missed + 1
    if covered+missed > 0:
        percent_covered = (covered*100.0)/(covered+missed)
    else:
        percent_covered = -1

    result = {'stockName':stockName, 'accuracy': test_acc  * 100,
    'positive_accuracy':positive_accuracy, 'percent_covered':percent_covered,
    'training_accuracy' : final_train_accuracy, 'prediction' : result_labels }

    return result
    #return test_acc / test_batches * 100

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
