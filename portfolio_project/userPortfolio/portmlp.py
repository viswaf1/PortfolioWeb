import datetime, sys, os, multiprocessing, time
from userPortfolio.models import UserTransactionsModel, UserPortfolioModel, AllStocksModel, SNP500Model
import csv, hashlib
import pandas, urllib.request, urllib.error, urllib.parse, csv, random, datetime, string, subprocess
from pytz import timezone
from dateutil.relativedelta import relativedelta
from math import pi
from talib.abstract import *
import talib
import userPortfolio.backend as backend
import numpy as np
from sklearn import svm
from sklearn.preprocessing import normalize
from scipy.ndimage.interpolation import shift

import theano
import theano.tensor as T

import lasagne

sys.path.append('libsvm/python')
from svmutil import *

sys.path.append('liblinear/python')
from liblinearutil import *

def run_lookahead_vs_accuracy():
    import numpy as np
    all_means = []
    for i in range(1,10):
        svc = SVMClassifier(num_days=200)
        svc.look_ahead = i
        ret = svc.future_test(test_period=1, num_periods=20, moveback=10)
        retarr = np.array(ret)
        all_means.append(np.mean(retarr))
    return all_means

def run_minprice_vs_accuracy():
    import numpy as np
    import matplotlib.pyplot as plt
    all_means = []
    for i in range(10, 410, 20):
        svc = SVMClassifier(num_days=200)
        svc.min_price = i
        ret = svc.future_test(test_period=1, num_periods=20, moveback=10)
        retarr = np.array(ret)
        all_means.append(np.mean(retarr))

    plt.plot(all_means)
    plt.show()
    return all_means

def run_numdays_vs_accuracy():
    import numpy as np
    import matplotlib.pyplot as plt
    all_means = []
    for i in range(100, 1001, 100):
        svc = SVMClassifier(num_days=i)
        ret = svc.future_test(test_period=1, num_periods=20, moveback=10)
        retarr = np.array(ret)
        all_means.append(np.mean(retarr))

    plt.plot(all_means)
    plt.show()

    return all_means

def run_gamma_vs_accuracy():
    import numpy as np
    import matplotlib.pyplot as plt
    all_means = []
    gammas = []
    i = 0.24
    while i <= 0.25:
        i = i+0.001
        gammas.append(i)
        svc = SVMClassifier(num_days=100)
        svc.gamma = i
        ret = svc.future_test(test_period=1, num_periods=20, moveback=10)
        retarr = np.array(ret)
        all_means.append(np.mean(retarr))

    fig, ax = plt.subplots()
    plt.plot(gammas, all_means, marker='o')
    ax.set_xticks(gammas)
    ax.set_yticks(all_means)
    plt.show()


    return all_means

def pick_and_run_top_stocks(top_percent=70):
    import operator
    svc = SVMClassifier(num_days=100)
    ret = svc.future_test(test_period=1, num_periods=111, moveback=10, period_start=1)
    retarr = np.array(ret)
    print(" Selection Mean "+str(np.mean(retarr)))
    # sucdic = svc.successes_dic
    # sorted_suc = sorted(sucdic.items(), key=operator.itemgetter(1))
    # num = (top_percent/100)*len(sorted_suc)
    # top_suc = sorted_suc[-num:]
    # top_stocks = [x[0] for x in top_suc]
    # faildic = svc.fails_dic
    # sorted_fail = sorted(faildic.items(), key=operator.itemgetter(1))
    # num = (top_percent/100)*len(sorted_suc)
    # top_fail = sorted_fail[-num:]
    # bad_stocks = [x[0] for x in top_fail]
    # import pdb; pdb.set_trace()
    # all_stocks = sucdic.keys()
    # selected_stocks = []
    # for eachStock in all_stocks:
    #     if eachStock not in bad_stocks:
    #         selected_stocks.append(eachStock)
    # selected_stocks = top_stocks
    #
    # print top_stocks


    sorted_results = sorted(list(svc.all_results.items()), key=operator.itemgetter(1))
    num = (top_percent/100)*len(sorted_results)
    top_results = sorted_results[-num:]
    selected_stocks = [x[0] for x in top_results]

    svc = SVMClassifier(num_days=100)
    ret = svc.future_test(test_period=1, num_periods=1, moveback=5,
    period_start=0, stockNames = selected_stocks)
    retarr = np.array(ret)
    print("Top Stocks Mean "+str(np.mean(retarr)))

    return svc


class MLPClassifier:
    def __init__(self, date='', type='snp500', num_days=100):
        self.date = date
        self.type = type
        self.look_ahead = 5
        self.min_price = 200
        self.gamma = 0.25
        self.C = 100
        self.num_days = num_days
        self.results = []
        self.labels = []

    def build_classifier(self):
        pass

    def async_callback(self, result):
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

            if data[i][-1] == np.nan:
                continue
            elif data[i][-1] < 0:
                labels.append(0)
            elif data[i][-1] > 0:
                labels.append(1)

            features.append(data[i][:-1].tolist())
        self.period_features.append({'stockName': stock_name, 'features':features, 'labels':labels})


    def future_test(self, test_period=100, num_periods=5, moveback = False, period_start = 0, stockNames = []):
        self.svm_result = []

        self.successes_dic = {}
        self.fails_dic = {}
        self.all_results = []
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
            self.period_features = []
            for stockName in stockNames:
                input = []
                try:
                    stock_data = backend.StockData.Instance().get_historical_stock_data(stockName)
                except Exception as err:
                    print("Error getting data for " + stockName + " " + str(err))
                    continue
                if len(stock_data) < 1:
                    continue
                if len(stock_data) > start_ind:
                    stock_slice = stock_data[-start_ind:-end_ind]
                    if stock_slice.iloc[-1].close > self.min_price:
                        period_stock_data.append((stockName, stock_slice, test_period+self.num_days+self.look_ahead, self.look_ahead, offset, False))

            pool = multiprocessing.Pool(5)
            for eachStockData in period_stock_data:
                pool.apply_async(get_feature_label_for_stocks_raw, args=(eachStockData,), callback=self.async_callback)
            pool.close()
            pool.join()

            period_train_test_data = []
            for eachFeature in self.period_features:
                features = eachFeature['features']
                labels = eachFeature['labels']

                train_data = features[:self.num_days]
                train_labels = labels[:self.num_days]
                test_data = features[self.num_days+self.look_ahead:self.num_days+self.look_ahead+test_period]
                test_labels = labels[self.num_days+self.look_ahead:self.num_days+self.look_ahead+test_period]
                period_train_test_data.append((train_data, train_labels, test_data, test_labels, 3, 500, eachFeature['stockName']))

            #self.svm_result = []
            pool = multiprocessing.Pool(2)
            for eachTrainTest in period_train_test_data:
                #pool.apply_async(run_MLP, args=(eachTrainTest,), callback=self.svm_async_callback)
                ret = run_MLP(eachTrainTest)
                self.svm_result.append(ret['positive_accuracy'])
                self.all_results.append(ret)
            pool.close()
            pool.join()

        #backend.StockData.Instance().flush()

        return self.svm_result


    def cointoss_async_callback(self, result):
        self.cointoss_results.append(result)


def run_MLP(xxx_todo_changeme):

    (train, train_labels, test, test_labels, depth, width, stockName) = xxx_todo_changeme
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
    print(("  test loss:\t\t\t{:.6f}".format(test_err)))
    print(("  test accuracy:\t\t{:.2f} %".format(
        test_acc * 100)))

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


def run_cointoss(xxx_todo_changeme1):
    (train, train_labels, test, test_labels, C) = xxx_todo_changeme1
    iter_results = []
    for i in range(len(test_labels)):
        if random.randint(1,2) == 1:
            iter_results.append(1)
        else:
            iter_results.append(0)
    iter_success = 0
    iter_fails = 0
    for x in range(len(test_labels)):
        if iter_results[x] > 0.1:
            if  test_labels[x] > 0.1:
                iter_success = iter_success + 1
            else:
                iter_fails = iter_fails + 1
    iter_outcome = ((iter_success)/(iter_success*1.0 + iter_fails*1.0))*100


    print((" Cointoss Accuracy =  : "+str(iter_outcome)+'% ('+str(iter_success)+" / "+str(iter_fails)+')'))
    result = iter_outcome
    # if iter_results[0] > 0.1:
    #     return(iter_outcome)
    # else:
    #     return(-1)

    if test_labels[0] > 0.1:
        return(iter_outcome)
    else:
        return(-1)


def get_feature_label_for_stocks_raw(xxx_todo_changeme2):
    #import pdb; pdb.set_trace()
    # data = data[:-200]
    # snp_data = snp_data[:-200]
    # nasdaq_data = nasdaq_data[:-200]
    (stock_name, data, num_days, look_ahead, offset, blind) = xxx_todo_changeme2
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



def get_feature_label_for_stocks(xxx_todo_changeme3):
    #import pdb; pdb.set_trace()
    # data = data[:-200]
    # snp_data = snp_data[:-200]
    # nasdaq_data = nasdaq_data[:-200]
    (stock_name, data, num_days, look_ahead, blind) = xxx_todo_changeme3
    start_time = time.time()
    offset = 40
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

    if data.shape[0] < slice_len:
        return ''

    try:
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
        ado = talib.ADOSC(high, low, data, volume)
        natr = talib.NATR(high, low, data)
        ultimate = talib.ULTOSC(high, low, data)

        ma5 = talib.MA(data, timeperiod=5)
        ma10 = talib.MA(data, timeperiod=10)
        disparity5 = data/ma5
        disparity10 = data/ma10
        beta = talib.BETA(high, low)
        adx = talib.ADX(high, low, data)

        index_mom = talib.MOM(index_data, timeperiod=5)
        index_disparity5 = index_data/talib.MA(index_data, timeperiod=5)
        index_macd, index_macd_signal, index_macd_hist = talib.MACD(index_data)
        index_disparity5 = index_disparity5/np.nanmax(np.abs(index_disparity5))
        index_mom = index_mom/np.nanmax(np.abs(index_mom))
        index_macd_hist = index_macd_hist/np.nanmax(np.abs(index_macd_hist))
        #nasdaq_mom1 = talib.MOM(nasdaq_data)

        mom10 = mom10/np.nanmax(np.abs(mom10))
        mom3 = mom3/np.nanmax(np.abs(mom3))
        willr10 = willr10/np.nanmax(np.abs(willr10))
        rsi16 = rsi16/np.nanmax(np.abs(rsi16))
        cci12 = cci12/np.nanmax(np.abs(cci12))
        rocr3 = rocr3/np.nanmax(np.abs(rocr3))
        macdhist = macdhist/np.nanmax(np.abs(macdhist))
        natr = natr/np.nanmax(np.abs(natr))
        adx = adx/np.nanmax(np.abs(adx))
        #nasdaq_mom1 = nasdaq_mom1/np.nanmax(np.abs(nasdaq_mom1))

        percentk = percentk/np.nanmax(np.abs(percentk))
        percentd = percentd/np.nanmax(np.abs(percentd))
        slowpercentd = slowpercentd/np.nanmax(np.abs(slowpercentd))
        ado = ado/np.nanmax(np.abs(ado))
        disparity5 = disparity5/np.nanmax(np.abs(disparity5))
        disparity10 = disparity10/np.nanmax(np.abs(disparity10))
        ultimate = ultimate/np.nanmax(np.abs(ultimate))
        beta = beta/np.nanmax(np.abs(beta))

        labels = get_label(data, look_ahead)


        #feature_matrix = np.column_stack((mom10, mom3, rsi16, cci12, macdhist, ado, willr10, disparity5, disparity10, beta, index_macd_hist, labels))
        feature_matrix = np.column_stack((mom10, cci12, percentd, beta, disparity10, index_disparity5, ultimate, adx, labels))
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
        print(str(e))
        feature_matrix = np.array([])
    return (stock_name, feature_matrix)



def get_feature_label_for_stocks_rdp(xxx_todo_changeme4):
    #import pdb; pdb.set_trace()
    # data = data[:-200]
    # snp_data = snp_data[:-200]
    # nasdaq_data = nasdaq_data[:-200]
    (stock_name, data, num_days, look_ahead) = xxx_todo_changeme4
    start_time = time.time()
    offset = 40
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

    if data.shape[0] < slice_len:
        return ''

    np.seterr(invalid='ignore')
    try:
        ema15 = talib.EMA(data, timeperiod=15)
        data_5 = shift(data, 5, cval=np.NaN)
        data_10 = shift(data, 10, cval=np.NaN)
        data_15 = shift(data, 15, cval=np.NaN)
        data_20 = shift(data, 20, cval=np.NaN)

        rdp5 = ((data-data_5)/(data_5))*100
        rdp10 = ((data-data_10)/(data_10))*100
        rdp15 = ((data-data_15)/(data_15))*100
        rdp20 = ((data-data_20)/(data_20))*100

        rdp5_std = np.nanstd(rdp5)
        rdp5[rdp5 > 2*rdp5_std] = 2*rdp5_std
        rdp5[rdp5 < -2*rdp5_std] = -2*rdp5_std

        rdp10_std = np.nanstd(rdp10)
        rdp10[rdp10 > 2*rdp10_std] = 2*rdp10_std
        rdp10[rdp10 < -2*rdp10_std] = -2*rdp10_std

        rdp15_std = np.nanstd(rdp15)
        rdp15[rdp15 > 2*rdp15_std] = 2*rdp15_std
        rdp15[rdp15 < -2*rdp15_std] = -2*rdp15_std

        rdp20_std = np.nanstd(rdp20)
        rdp20[rdp20 > 2*rdp20_std] = 2*rdp20_std
        rdp20[rdp20 < -2*rdp20_std] = -2*rdp20_std

        #Append required technical indicators to the data
        ema15 = ema15/np.nanmax(np.abs(ema15))
        rdp5 = rdp5/np.nanmax(np.abs(rdp5))
        rdp10 = rdp10/np.nanmax(np.abs(rdp10))
        rdp15 = rdp15/np.nanmax(np.abs(rdp15))
        rdp20 = rdp20/np.nanmax(np.abs(rdp20))
        labels = get_label(data, look_ahead)

        feature_matrix = np.column_stack((ema15, rdp5, rdp10, rdp15, rdp20, labels))


        #feature_matrix = np.column_stack((mom10, mom3, rsi16, cci12, macdhist, percentk, percentd, ado, willr10, disparity5, disparity10, labels))
        #print("--- %s seconds ---" % (time.time() - start_time))
        feature_matrix = feature_matrix[offset:-(look_ahead)]
    except Exception as e:
        print(str(e))
        feature_matrix = np.array([])
    return (stock_name, feature_matrix)


# def get_label(data, look_ahead):
#     #data - numpy array
#     labels = np.array([])
#     data_len = data.size
#     for i in range(data_len):
#         if i+look_ahead >= data_len:
#             labels = np.append(labels, np.nan)
#             continue
#         sma = 0
#         for s in range(1,look_ahead+1):
#             sma = sma + data[i+s]
#         sma = sma/look_ahead
#         if sma > data[i]:
#             labels = np.append(labels, 1)
#         else:
#             labels = np.append(labels, -1)
#
#     return labels

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
            if change > 0 and (change/data[i])*100 >= 5.0:
                found = True
                labels = np.append(labels, 1)
                break
        if not found:
            labels = np.append(labels, -1)

    return labels

def get_label2(data, look_ahead):
    #data - numpy array
    cost = 10
    tax = 30
    money = 1000
    labels = np.array([])
    data_len = data.size
    for i in range(data_len):
        if i+look_ahead >= data_len:
            labels = np.append(labels, np.nan)
            continue
        buyPrice = data[i]
        ad = 0
        found = False
        num_stocks = money/buyPrice
        for x in range(look_ahead):
            ad = ad + 1
            current_data = data[i+ad]
            currentPrice = current_data
            if currentPrice > buyPrice:
                diff = (currentPrice-buyPrice)*num_stocks
                profit = diff - (0.3*diff)
                if profit >= cost:
                    found = True
                    break
        if found:
            labels = np.append(labels, 1)
        else:
            labels = np.append(labels, -1)

    return labels

#needs current stock data, snp500 data (^GSPC), NASDAQ data (^IXIC)
def get_feature_vector(data, snp_data, nasdaq_data, date):
    loc = data.index.get_loc(date)
    snp_loc = snp_data.index.get_loc(date)
    nasdaq_loc = nasdaq_data.index.get_loc(date)
    data_slice = data[:date].tail(40)

    mom1 = data.iloc[loc].close - data.iloc[loc-1].close
    mom3 = data.iloc[loc].close - data.iloc[loc-3].close
    willr10 = (WILLR(data_slice)).iloc[-1]
    rsi16 = (RSI(data_slice, timeperiod=16)).iloc[-1]
    cci12 = (CCI(data_slice, timeperiod=12)).iloc[-1]
    rocr3 = (ROCR(data_slice, timeperiod=3)).iloc[-1]
    macd_data = MACD(data_slice)
    macdhist = (macd_data.iloc[-1]).macdhist

    nasdaq_mom1 = nasdaq_data.iloc[nasdaq_loc].close - nasdaq_data.iloc[nasdaq_loc-1].close
    snp_mom1 = snp_data.iloc[snp_loc].close - snp_data.iloc[snp_loc-1].close

    feature_vec = [mom1, rsi16, willr10, nasdaq_mom1, mom1, cci12, snp_mom1, mom3, rocr3, macdhist]
    return np.array(feature_vec)


def compare_test_periods(num_periods = 1):
    results = []
    for i in range(1,10):
        svc = SVMClassifier(num_days=100)
        ret = svc.future_test(test_period=i, num_periods=num_periods, moveback=100)
        acc = 0
        for each in ret:
            acc = acc + each
        acc = acc/len(ret)
        results.append(acc)
    return results


#
