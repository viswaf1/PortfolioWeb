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

sys.path.append('libsvm/python')
from svmutil import *

class EnsambleClassifier:
    def __init__(self, date='', type='forex', num_days=100):
        self.date = date
        self.type = type
        self.look_ahead = 5
        self.min_price = 0
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

        for i in range(len(data)):
            if np.isnan(data[i][-1]):
                labels.append(np.nan)
            elif data[i][-1] < 0:
                labels.append(-1)
            elif data[i][-1] > 0:
                labels.append(1)

        for i in range(len(data)):
            if np.isnan(data[i][-2]):
                negative_labels.append(np.nan)
            elif data[i][-2] < 0:
                negative_labels.append(-1)
            elif data[i][-2] > 0:
                negative_labels.append(1)

            features.append(data[i][:-2].tolist())
        self.period_features_lock.acquire()
        if stock_name in self.period_features:
            self.period_features[stock_name]['svm'] = {'stockName': stock_name, 'features':features, 'labels':labels}
            self.period_features[stock_name]['svm_negative'] = {'stockName': stock_name, 'features':features, 'labels':negative_labels}
        else:
            self.period_features[stock_name] = {'svm' :{'stockName': stock_name, 'features':features, 'labels':labels}}
            self.period_features[stock_name]['svm_negative'] = {'stockName': stock_name, 'features':features, 'labels':negative_labels}
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
            if self.type == 'forex':
                stocks = USDForexModel.objects.all()
            stockNames = []
            for eachStock in stocks:
                stockNames.append(eachStock.stockName)


        #append close_snp to the stock data
        all_inputs = []
        #backend.StockData.Instance().append_stock_column('^GSPC', 'close', '_index')
        backend.StockData.Instance().offline = True
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
            pool.apply_async(get_feature_label_for_stocks, args=(eachStockData,), callback=self.async_callback_svm_features)
            pool.apply_async(get_feature_label_for_stocks_raw, args=(eachStockData,), callback=self.async_callback_mlp_features)
        pool.close()
        pool.join()
        period_train_test_data = []
        pf_keys = self.period_features.keys()
        for eachKey in pf_keys:
            eachFeature = self.period_features[eachKey]
            fkeys = eachFeature.keys()
            combined = {}
            for fk in fkeys:
                features = eachFeature[fk]['features']
                labels = eachFeature[fk]['labels']

                train_data = features[:self.num_days]
                train_labels = labels[:self.num_days]
                test_data = [features[-1]]
                test_labels = [1.0]
                if(fk == 'svm' or fk == 'svm_negative'):
                    combined[fk] = (train_data, train_labels, test_data, test_labels, self.C, self.gamma, self.epsilon, eachFeature[fk]['stockName'])
                else:
                    combined[fk] = (train_data, train_labels, test_data, test_labels, 3, 100, eachFeature[fk]['stockName'])
            period_train_test_data.append(combined)

        self.combined_result = []
        for eachTrainTest in period_train_test_data:
            svm_ret = run_libSVMGPU(eachTrainTest['svm'])
            svm_neg_ret = run_libSVMGPU(eachTrainTest['svm_negative'])
            #mlp_ret = run_MLP(eachTrainTest['mlp'])
            test_labels = eachTrainTest['mlp'][3]
            poss = []
            svm_labels = svm_ret['prediction']
            #mlp_labels = mlp_ret['prediction']
            for i in range(len(test_labels)):
                #temp_dic = {'svm_ret':svm_ret, 'mlp_ret':mlp_ret, 'svm_neg_ret':svm_neg_ret}
                temp_dic = {'svm_ret':svm_ret, 'svm_neg_ret':svm_neg_ret}
                #temp_dic = {'mlp_ret':mlp_ret}
                temp_dic['actual'] = test_labels[i]
                self.combined_result.append(temp_dic)
        #picked_stocks = [x['svm_ret']['stockName'] for x in self.combined_result if x['svm_ret']['prediction'][0] > 0.1 and x['mlp_ret']['prediction'][0] > 0.1 and x['mlp_ret']['training_accuracy'] > 0.59 and x['mlp_ret']['training_accuracy'] < 0.69 and x['svm_ret']['training_accuracy'] < 0.60 and x['svm_ret']['training_accuracy'] > 0.30]
        #picked_stocks = [x['svm_ret']['stockName'] for x in self.combined_result if x['svm_ret']['prediction'][0] > 0.1 and x['mlp_ret']['prediction'][0] > 0.1 and x['svm_neg_ret']['prediction'][0] > 0.1]
        picked_stocks = [x['svm_ret']['stockName'] for x in self.combined_result if x['svm_ret']['prediction'][0] > 0.1 and x['svm_neg_ret']['prediction'][0] > 0.1]
        print picked_stocks
        return picked_stocks
        #return self.rank_picks(picked_stocks, pick_date)

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
                pool.apply_async(get_feature_label_for_stocks, args=(eachStockData,), callback=self.async_callback_svm_features)
                pool.apply_async(get_feature_label_for_stocks_raw, args=(eachStockData,), callback=self.async_callback_mlp_features)
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
                        combined[fk] = (train_data, train_labels, test_data, test_labels, 4, 500, eachFeature[fk]['stockName'])
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

    train_loss, final_train_accuracy = val_fn(X_train, y_train)
    test_err, test_acc = val_fn(X_test, y_test)
    test_err = 0+test_err
    test_acc = 0+test_acc
    # print("Final results:")
    # print("  test loss:\t\t\t{:.6f}".format(test_err))
    # print("  test accuracy:\t\t{:.2f} %".format(
    #     test_acc * 100))

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
    data = data_frame['close'].values
    high = data_frame['high'].values
    low = data_frame['low'].values
    volume = data_frame['volume'].values


    try:
        close = data/abs(np.mean(data))
        close = np.clip(close, -0.99, 0.99)

        high = high/abs(np.mean(high))
        high = np.clip(high, -0.99, 0.99)

        low = low/abs(np.mean(low))
        low = np.clip(low, -0.99, 0.99)

        volume = volume/abs(np.mean(volume))
        volume = np.clip(volume, -0.99, 0.99)

        labels = get_label(data, look_ahead)


        #feature_matrix = np.column_stack((mom10, mom3, rsi16, cci12, macdhist, ado, willr10, disparity5, disparity10, beta, index_macd_hist, labels))
        feature_matrix = np.column_stack((close, high, low,
        shift(close, 1, cval=np.NaN), shift(high, 1, cval=np.NaN), shift(low, 1, cval=np.NaN),
        shift(close, 2, cval=np.NaN), shift(high, 2, cval=np.NaN), shift(low, 2, cval=np.NaN),
        shift(close, 3, cval=np.NaN), shift(high, 3, cval=np.NaN), shift(low, 3, cval=np.NaN),
        shift(close, 4, cval=np.NaN), shift(high, 4, cval=np.NaN), shift(low, 4, cval=np.NaN),

        shift(close, 5, cval=np.NaN), shift(high, 5, cval=np.NaN), shift(low, 5, cval=np.NaN),
        shift(close, 6, cval=np.NaN), shift(high, 6, cval=np.NaN), shift(low, 6, cval=np.NaN),
        shift(close, 7, cval=np.NaN), shift(high, 7, cval=np.NaN), shift(low, 7, cval=np.NaN),
        shift(close, 8, cval=np.NaN), shift(high, 8, cval=np.NaN), shift(low, 8, cval=np.NaN),

        labels))
        if not blind:
            feature_matrix = feature_matrix[offset:-(look_ahead)]
        else:
            feature_matrix = feature_matrix[offset:]
    except Exception as e:
        print ("portmlp: get_feature_label_for_stocks_raw : " + str(e))
        feature_matrix = np.array([])
    return (stock_name, feature_matrix)



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
    #index_data = data_frame['close_index'].values
    data = data_frame['close'].values
    high = data_frame['high'].values
    low = data_frame['low'].values

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

        #index_disparity5 = index_data/talib.MA(index_data, timeperiod=5)

        #index_disparity5 = index_disparity5/abs(np.nanmean(index_disparity5))

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
        #index_disparity5 = np.clip(index_disparity5, -1, 1)
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
        negative_labels = get_negative_label(data, look_ahead)
        #feature_matrix = np.column_stack((mom10, mom3, rsi16, cci12, macdhist, ado, willr10, disparity5, disparity10, beta, index_macd_hist, labels))
        #*feature_matrix = np.column_stack((mom10, cci12, percentd, beta, disparity10, index_disparity5, ultimate, adx, labels))
        feature_matrix = np.column_stack((mom10, cci12, percentd, beta, disparity10, ultimate, adx, labels, negative_labels))
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
        sma = 0
        found = False
        for s in range(1,look_ahead+1):
            change = data[i+s]-data[i]
            if change > 0 and (change/data[i])*100 >= 0.4:
                found = True
                labels = np.append(labels, 1)
                break
        # change = data[i+look_ahead] - data[i]
        # if change > 0 and (change/data[i])*100 >= 1.0:
        #     found = True
        #     labels = np.append(labels, 1)
        if not found:
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
        # change = data[i+look_ahead] - data[i]
        # if change > 0 and (change/data[i])*100 >= 1.0:
        #     found = True
        #     labels = np.append(labels, 1)
        if not found:
            labels = np.append(labels, 1)

    return labels

# #one if price increases by 5% in look ahead
# def get_label(data, look_ahead):
#     #data - numpy array
#     labels = np.array([])
#     data_len = data.size
#     for i in range(data_len):
#         if i+look_ahead >= data_len:
#             labels = np.append(labels, np.nan)
#             continue
#         sma = 0
#         found = False
#         for s in range(1,look_ahead+1):
#             change = data[i+s]-data[i]
#             if data[i+s] < data[i]:
#                 found = True
#                 labels = np.append(labels, -1)
#                 break
#         if not found:
#             labels = np.append(labels, 1)
#
#     return labels
