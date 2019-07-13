import datetime, sys, os, multiprocessing, time, cPickle
import threading
from userPortfolio.models import UserTransactionsModel, UserPortfolioModel, AllStocksModel, SNP500Model, USDForexModel
import csv, hashlib, itertools
import pandas, urllib2, csv, random, datetime, string, subprocess
from pytz import timezone
from dateutil.relativedelta import relativedelta
from math import pi
import userPortfolio.backend as backend
import numpy as np
from sklearn import svm
from scipy.ndimage.interpolation import shift
from operator import itemgetter
import talib
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt


def append_stock_data(data, append_data, append_data_name):
    data.merge(append_data, how='inner', suffixes=['', append_data_name]


class EnsambleModelBuilder:
    def __init__(self,  end_date='', type='snp500', num_days=500):
        self.end_date = date
        self.num_days = num_days
        self.type = type

        self.offset = 50
        self.test_period = 1

        self.look_ahead = 10
        self.min_price = 100
        self.gamma = 1.3
        self.C = 500
        self.epsilon = 0.001
        self.data_dir = "pipeline_data"

    def async_callback_svm_features(self, result):
        if result == '':
            return
        stock_name, features, labels = result
        #print stock_name
        if features.shape[0] < self.num_days:
            print "Error in stock data: "+stock_name
            return

        self.period_features_lock.acquire()
        if stock_name in self.period_features:
            self.period_features[stock_name]['svm'] = {'stockName': stock_name, 'features':features, 'labels':labels}
        else:
            self.period_features[stock_name] = {'svm' :{'stockName': stock_name, 'features':features, 'labels':labels}}
        self.period_features_lock.release()


    def test_model(self):

        time_slice = self.look_ahead+test_period+self.num_days+self.offset

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
            pool.apply_async(get_feature_label_for_stocks, args=(eachStockData,), callback=self.async_callback_svm_features)
        pool.close()
        pool.join()


def run_libSVMGPU((train, train_labels, test, test_labels, C, gamma, epsilon, stockName)):
    ram_disk = "/tmp/port_ramdisk/"
    suffix = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(10))
    filename = "svm_train_input"+suffix
    filepath = ram_disk+filename
    output_filename = "svm_model_out"+suffix
    output_filepath = ram_disk+output_filename
    with open(filepath, 'w') as input_file:
        for i in range(len(train)):
            line = str(train_labels[i])
            features = train[i]
            for j in range(len(features)):
                line = line + ' ' + str(j)+':'+str(features[j])
            line = line + '\n'

            input_file.write(line)
        input_file.close()


    proc = subprocess.Popen(['libsvm_withgpu/svm-train-gpu', '-c', str(C), '-g', str(gamma), '-e', str(epsilon), filepath, output_filepath],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    retcode = proc.wait()
    (stdoutdata, stderrdata) = proc.communicate()
    print(stdoutdata)
    if stderrdata:
        print(stderrdata)
    if retcode > 0:
        os.remove(filepath)
        print("Error "+str(stderrdata))
        return(-1)

    try:
        svm_model = svm_load_model(output_filepath)
        result_labels, accuracy, values = svm_predict(test_labels, test, svm_model)
        tempy, train_accuracy, v = svm_predict(train_labels, train, svm_model)

        os.remove(filepath)
        os.remove(output_filepath)
    except Exception as e:
        print(str(e))
        return(-1)

    positive_accuracy = 0
    pos_success = 0
    pos_fails = 0
    suceess = 0
    fails = 0
    for i in range(len(result_labels)):
        if result_labels[i] > 0.1:
            if test_labels[i] > 0.1:
                pos_success = pos_success + 1
            else:
                pos_fails = pos_fails + 1


    total_accuracy = accuracy[0]

    if pos_success+pos_fails > 0:
        positive_accuracy = (pos_success*100.0)/(pos_success+pos_fails)
    else:
        positive_accuracy = -1

    covered = 0
    missed = 0
    for i in range(len(test_labels)):
        if test_labels[i] > 0.1:
            if result_labels[i] > 0.1:
                covered = covered + 1
            else:
                missed = missed + 1
    if covered+missed > 0:
        percent_covered = (covered*100.0)/(covered+missed)
    else:
        percent_covered = -1
    result = {'stockName':stockName, 'accuracy': total_accuracy,
    'positive_accuracy':positive_accuracy, 'percent_covered':percent_covered,
    'prediction' : result_labels, 'training_accuracy' : train_accuracy[0]/100}

    return result



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

        labels_price_increase = get_label_price_increase(data, look_ahead)
        labels_no_price_increase = get_label_price_no_increase(data, look_ahead)
        labels_price_decrease = get_label_price_decrease(data, look_ahead)
        all_labels = np.column_stack((labels_price_increase, labels_no_price_increase, labels_price_decrease))

        feature_matrix = np.column_stack((mom10, cci12, percentd, beta, disparity10, index_disparity5, ultimate, adx, labels))
        if not blind:
            feature_matrix = feature_matrix[offset:-(look_ahead)]
        else:
            feature_matrix = feature_matrix[offset:]
    # except Exception as e:
    #     print ("portsvm:get_feature_label_for_stocks :" + str(e))
    #     feature_matrix = np.array([])
    return (stock_name, feature_matrix, all_labels)






#one if price increases by 5% in look ahead
def get_label_price_increase(data, look_ahead):
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
            if change > 0 and (change/data[i])*100 >= 5.0:
                found = True
                labels = np.append(labels, 1)
                break
        if not found:
            labels = np.append(labels, -1)

    return labels

#one if price increases by 5% in look ahead
def get_label_price_no_increase(data, look_ahead):
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
            if (change/data[i])*100 <= 0.5:
                found = True
                labels = np.append(labels, -1)
                break
        if not found:
            labels = np.append(labels, 1)

    return labels


#one if price increases by 5% in look ahead
def get_label_price_decrease(data, look_ahead):
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
            if (change/data[i])*100 <= 2.0:
                found = True
                labels = np.append(labels, -1)
                break
        if not found:
            labels = np.append(labels, 1)

    return labels


#
