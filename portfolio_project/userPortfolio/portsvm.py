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
from scipy.ndimage.interpolation import shift

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
        svc = SVMClassifier(num_days=250)
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
    # i = 0.24
    # while i <= 0.25:
    #     i = i+0.001
    i = 0.5
    while i <= 2:
        i = i+0.2
        gammas.append(i)
        svc = SVMClassifier(num_days=250)
        svc.gamma = i
        ret = svc.future_test(test_period=1, num_periods=200, moveback=10)
        retarr = np.array(ret)
        all_means.append(np.mean(retarr))

    fig, ax = plt.subplots()
    plt.plot(gammas, all_means, marker='o')
    ax.set_xticks(gammas)
    ax.set_yticks(all_means)
    plt.show()
    return all_means

def run_C_vs_accuracy():
    import numpy as np
    import matplotlib.pyplot as plt
    all_means = []
    gammas = []
    # i = 0.24
    # while i <= 0.25:
    #     i = i+0.001
    i = 400
    while i <= 600:
        i = i+20
        gammas.append(i)
        svc = SVMClassifier(num_days=250)
        svc.C = i
        ret = svc.future_test(test_period=1, num_periods=200, moveback=10)
        retarr = np.array(ret)
        all_means.append(np.mean(retarr))

    fig, ax = plt.subplots()
    plt.plot(gammas, all_means, marker='o')
    ax.set_xticks(gammas)
    ax.set_yticks(all_means)
    plt.show()
    return all_means

def run_epsilon_vs_accuracy():
    import numpy as np
    import matplotlib.pyplot as plt
    all_means = []
    gammas = []
    # i = 0.24
    # while i <= 0.25:
    #     i = i+0.001
    i = 0.000001
    while i <= 0.001:
        i = i*10
        gammas.append(i)
        svc = SVMClassifier(num_days=250)
        svc.epsilon = i
        ret = svc.future_test(test_period=1, num_periods=200, moveback=10)
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


class SVMClassifier:
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
                labels.append(-1)
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

            pool = multiprocessing.Pool(16)
            for eachStockData in period_stock_data:
                #return get_feature_label_for_stocks(eachStockData)
                pool.apply_async(get_feature_label_for_stocks, args=(eachStockData,), callback=self.async_callback)
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
                period_train_test_data.append((train_data, train_labels, test_data, test_labels, self.C, self.gamma, self.epsilon, eachFeature['stockName']))

            #self.svm_result = []
            pool = multiprocessing.Pool(16)
            for eachTrainTest in period_train_test_data:
                pool.apply_async(run_libSVMGPU, args=(eachTrainTest,), callback=self.svm_async_callback)
                #pool.apply_async(run_cointoss, args=(eachTrainTest,), callback=self.svm_async_callback)
                #ret = run_libSVMGPU(eachTrainTest)
            pool.close()
            pool.join()

        #backend.StockData.Instance().flush()

        return self.svm_result

    def svm_async_callback(self, result):
        # self.svm_result.append(result)
        self.all_results.append(result)
        # stockName = result['stockName']
        # if result['accuracy'] >= 0:
        #     if result['accuracy'] == 0:
        #         x = -1
        #     if result['accuracy'] > 0:
        #         x = 1
        #
        #     if stockName in self.all_results:
        #         self.all_results[stockName] = self.all_results[stockName] + x
        #     else:
        #         self.all_results[stockName] = x
        #
        #
        # if result['positive_accuracy'] >= 0:
        #     self.svm_result.append(result['positive_accuracy'])
        #
        #     if result['positive_accuracy'] > 0:
        #         if stockName in self.successes_dic:
        #             self.successes_dic[stockName] = self.successes_dic[stockName]+1
        #         else:
        #             self.successes_dic[stockName] = 1
        #     else:
        #         if stockName in self.fails_dic:
        #             self.fails_dic[stockName] = self.fails_dic[stockName]+1
        #         else:
        #             self.fails_dic[stockName] = 1
        #
        # if result['percent_covered'] >= 0:
        #     self.percent_covered.append(result['percent_covered'])


    def cointoss_async_callback(self, result):
        self.cointoss_results.append(result)

def run_libSVMGPU(xxx_todo_changeme):
    (train, train_labels, test, test_labels, C, gamma, epsilon, stockName) = xxx_todo_changeme
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
        print(("Error "+str(stderrdata)))
        return(-1)

    try:
        svm_model = svm_load_model(output_filepath)
        result_labels, accuracy, values = svm_predict(test_labels, test, svm_model)
        tempy, train_accuracy, v = svm_predict(train_labels, train, svm_model)

        os.remove(filepath)
        os.remove(output_filepath)
    except Exception as e:
        print((str(e)))
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



def run_libSVM(xxx_todo_changeme1):
    (train, train_labels, test, test_labels, C) = xxx_todo_changeme1
    svm_prob = svm_problem(train_labels, train, isKernel=True)
    param = svm_parameter('-t 2 -c %d -b 0 -h 0' % (C))
    model = svm_train(svm_prob, param)

    import datetime
    model_name = str(len(train))+'-'+str(datetime.datetime.now())
    #svm_save_model(model_name, model)

    result_labels, accuracy, values = svm_predict(test_labels, test, model)

    print(accuracy)
    if result_labels[0] > 0.1:
        return(accuracy[0])
    else:
        return(-1)
    # if test_labels[0] > 0.1:
    #     return(accuracy[0])
    # else:
    #     return(-1)

def run_linearSVM(xxx_todo_changeme2):
    (train_data, train_labels, test_data, test_labels, C) = xxx_todo_changeme2
    svm_prob = problem(train_labels, train_data)
    param = parameter('-s 2 -c %d -n 8' % (C))
    model = train(svm_prob, param)

    import datetime
    model_name = str(len(train_data))+'-'+str(datetime.datetime.now())

    result_labels, accuracy, values = predict(test_labels, test_data, model)

    # successes = 0.0
    # fails = 0.0
    # for x in range(len(test_labels)):
    #     if result_labels[x] > 0.1:
    #         if test_labels[x] > 0.1:
    #             successes = successes + 1
    #         else:
    #             fails = fails + 1
    # pos_accuracy = (successes*100.0)/(successes+fails)
    #
    # print(str(result_labels))
    print(accuracy)
    print(("Positive Accuracy : " +str(pos_accuracy) + " ("+str(successes)+"/"+str(fails)+")"))
    if result_labels[0] > 0.1:
        return(accuracy[0])
    else:
        return(-1)
    # if test_labels[0] > 0.1:
    #     return(accuracy[0])
    # else:
    #     return(-1)


def run_SVC(xxx_todo_changeme3):
    (train, train_labels, test, test_labels, C) = xxx_todo_changeme3
    clf = svm.SVC(C=C)
    clf.fit(train, train_labels)

    iter_results = clf.predict(test)
    iter_success = 0
    iter_fails = 0
    false_positive = 0
    false_negative = 0
    missed = 0
    for x in range(len(test_labels)):
        if test_labels[x] > 0.1:
            if iter_results[x] <= 0.1:
                missed = missed + 1
        if iter_results[x] > 0.1:
            if test_labels[x] > 0.1:
                iter_success = iter_success + 1
            else:
                iter_fails = iter_fails + 1


        # if iter_results[x] == test_labels[x]:
        #     iter_success = iter_success + 1
        # else:
        #     iter_fails = iter_fails + 1
        #     if iter_results[x] > 0:
        #         false_positive = false_positive + 1
        #     else:
        #         false_negative = false_negative + 1
    iter_outcome = (iter_success)/(iter_success*1.0 + iter_fails*1.0)
    print((" C Success/Failure : "+str(C)+" : "+str(iter_success)+" / "+str(iter_fails)+" , Missed "+str(missed)+' , Total '+str(len(test_labels)) +
    ' Percent classified as 1 :'+str((float(iter_success+iter_fails)/len(test_labels))*100) ))
    result = iter_outcome
    return result


def run_cointoss(xxx_todo_changeme4):
    (train, train_labels, test, test_labels, C, gamma, epsilon, stockName) = xxx_todo_changeme4
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

    result = {'stockName':stockName, 'accuracy': 0,
    'positive_accuracy':iter_outcome, 'percent_covered':0,
    'prediction' : [0], 'training_accuracy' : 0}

    return result



def get_feature_label_for_stocks(xxx_todo_changeme5):
    #import pdb; pdb.set_trace()
    # data = data[:-200]
    # snp_data = snp_data[:-200]
    # nasdaq_data = nasdaq_data[:-200]
    (stock_name, data, num_days, look_ahead, offset, blind) = xxx_todo_changeme5
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

        #feature_matrix = np.column_stack((mom10, mom3, rsi16, cci12, macdhist, ado, willr10, disparity5, disparity10, beta, index_macd_hist, labels))
        #*feature_matrix = np.column_stack((mom10, cci12, percentd, beta, disparity10, index_disparity5, ultimate, adx, labels))
        feature_matrix = np.column_stack((mom10, cci12, percentd, beta, disparity10, index_disparity5, ultimate, adx, labels))
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

def get_feature_label_for_stocks_raw(xxx_todo_changeme6):
    #import pdb; pdb.set_trace()
    # data = data[:-200]
    # snp_data = snp_data[:-200]
    # nasdaq_data = nasdaq_data[:-200]
    (stock_name, data, num_days, look_ahead) = xxx_todo_changeme6
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
        feature_matrix = feature_matrix[offset:-(look_ahead)]
    except Exception as e:
        print(str(e))
        feature_matrix = np.array([])
    return (stock_name, feature_matrix)


#
