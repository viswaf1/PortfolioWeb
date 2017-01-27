import datetime, sys, os, multiprocessing, time
from userPortfolio.models import UserTransactionsModel, UserPortfolioModel, AllStocksModel, SNP500Model
import csv, hashlib
import pandas, urllib2, csv, random
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

os.environ["OMP_NUM_THREADS"] = "8"

class SVMClassifier:
    def __init__(self, date='', type='snp500', num_days=100):
        self.date = date
        self.type = type
        self.look_ahead = 5
        self.num_days = num_days
        self.results = []
        self.labels = []
        for i in range(self.num_days):
            self.results.append([])
            self.labels.append([])

    def build_classifier(self):
        pass

    def async_callback(self, result):
        if result == '':
            return
        stock_name, data = result
        #print stock_name
        if data.shape[0] < self.num_days:
            print "Error in stock data: "+stock_name
            return
        for i in range(len(data)):
            if len(self.results) <= i:
                self.results.append([])
            if len(self.labels) <= i:
                self.labels.append([])
            if data[i][-1] == np.nan:
                continue
            elif data[i][-1] < 0:
                self.labels[i].append(0)
            elif data[i][-1] > 0:
                self.labels[i].append(1)

            self.results[i].append(data[i][:-1].tolist())


    def future_test(self, test_period=100, num_periods=5, moveback = False):

        all_inputs = []
        self.svm_result = []
        for period in range(num_periods):
            #train for number of days and test for test_period
            offset = 50
            slice = (2*self.look_ahead)+test_period+self.num_days+offset

            if moveback:
                start_ind = ((period+1)*moveback)+slice
                end_ind = ((period)*moveback)+1
            else:
                start_ind = (period+1)*slice
                end_ind = (period)*slice+1

            if self.type == 'allstocks':
                stocks = AllStocksModel.objects.all()
            elif self.type == 'snp500':
                stocks = SNP500Model.objects.all()

            input = []
            #append close_snp to the stock data
            backend.StockData.Instance().append_stock_column('^GSPC', 'close', '_index')
            for eachStock in stocks:
                stockName = eachStock.stockName
                try:
                    stock_data = backend.StockData.Instance().get_historical_stock_data(stockName)
                except Exception as err:
                    print "Error getting data for " + stockName + " " + str(err)
                    continue
                if len(stock_data) < 1:
                    continue
                if len(stock_data) > start_ind:
                    stock_slice = stock_data[-start_ind:-end_ind]
                    if stock_slice.iloc[-1].close > 20:
                        input.append((stockName, stock_slice, test_period+self.num_days+self.look_ahead, self.look_ahead))

            self.results = []
            self.labels = []
            for i in range(self.num_days):
                self.results.append([])
                self.labels.append([])
            pool = multiprocessing.Pool(20)
            for eachInput in input:
                pool.apply_async(get_feature_label_for_stocks_rdp, args=(eachInput,), callback=self.async_callback)
                #get_feature_label_for_stocks_rdp(eachInput)
            pool.close()
            pool.join()
            input = []
            temp_train_data = self.results[:self.num_days]
            temp_train_labels = self.labels[:self.num_days]
            temp_test_data = self.results[self.num_days+self.look_ahead:self.num_days+self.look_ahead+test_period]
            temp_test_labels = self.labels[self.num_days+self.look_ahead:self.num_days+self.look_ahead+test_period]
            #print('Test start index : '+str(self.num_days+self.look_ahead)+' End index : '+str(self.num_days+self.look_ahead+test_period))
            train_data = []
            train_labels = []
            test_data = []
            test_labels = []

            out_list = [train_data, train_labels, test_data, test_labels]
            in_list = [temp_train_data, temp_train_labels, temp_test_data, temp_test_labels]

            for ind in range(len(in_list)):
                inp = in_list[ind]
                out = out_list[ind]
                for each in inp:
                    for eachIn in each:
                        out.append(eachIn)
                in_list[ind] = []
            #return train_data
            print("train size: "+str(len(train_data))+" train label size: "+ str(len(train_labels)) + " test size : "+str(len(test_data))+" test label size: "+str(len(test_labels)))
            input = (train_data, train_labels, test_data, test_labels, 100)
            #all_inputs.append(input)
            ret = run_linearSVM(input)
            run_cointoss(input)
            self.svm_result.append(ret)
        return train_labels
        backend.StockData.Instance().flush()

        # self.svm_result = []
        # #ool = multiprocessing.Pool(8)
        # for x in range(len(all_inputs)):
        #     eachInput = all_inputs[x]
        #     #pool.apply_async(run_SVC, args=(eachInput,), callback=self.svm_async_callback)
        #     #pool.apply_async(run_libSVM, args=(eachInput,), callback=self.svm_async_callback)
        #     #ret = run_libSVM(eachInput)
        #     ret = run_linearSVM(eachInput)
        #     run_cointoss(eachInput)
        #     self.svm_result.append(ret)
        #     all_inputs[x] = []
        # #pool.close()
        # #pool.join()
        return self.svm_result

    def verify_with_model(self, model_name, test_period=10):

        if self.type == 'allstocks':
            stocks = AllStocksModel.objects.all()
        elif self.type == 'snp500':
            stocks = SNP500Model.objects.all()

        input = []
        for eachStock in stocks:
            stockName = eachStock.stockName
            try:
                stock_data = backend.StockData.Instance().get_historical_stock_data(stockName)
            except Exception as err:
                print "Error getting data for " + stockName + " " + str(err)
                continue
            if len(stock_data) < 1:
                continue
            if len(stock_data) > test_period+100:
                stock_slice = stock_data[: -(test_period+100)]
                if stock_slice.iloc[-1].close > 100:
                    input.append((stockName, stock_slice, test_period, self.look_ahead))


        self.results = []
        self.labels = []
        for i in range(self.num_days):
            self.results.append([])
            self.labels.append([])
        pool = multiprocessing.Pool(20)
        for eachInput in input:
            #pool.apply_async(get_feature_label_for_stocks, args=(eachInput,), callback=self.async_callback)
            result = get_feature_label_for_stocks(eachInput)
            if result == '':
                continue
            print result
            stock_name, data = result
            #print stock_name
            if data.shape[0] < self.num_days:
                print "Error in stock data: "+stock_name
                continue
            for i in range(len(data)):
                if len(self.results) <= i:
                    self.results.append([])
                if len(self.labels) <= i:
                    self.labels.append([])
                if data[i][-1] == np.nan:
                    continue
                elif data[i][-1] < 0:
                    self.labels[i].append(0)
                elif data[i][-1] > 0:
                    self.labels[i].append(1)

                self.results[i].append(data[i][:-1].tolist())
        pool.close()
        pool.join()

        temp_test_data = self.results
        temp_test_labels = self.labels

        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        out_list = [test_data, test_labels]
        in_list = [temp_test_data, temp_test_labels]
        for ind in range(len(in_list)):
            inp = in_list[ind]
            out = out_list[ind]
            for each in inp:
                for eachIn in each:
                    out.append(eachIn)
        print("test size : "+str(len(test_data))+" test label size: "+str(len(test_labels)))

        model = svm_load_model(model_name)
        result_labels, accuracy, values = svm_predict(test_labels, test_data, model)

        print(accuracy)



    def cross_validate(self, k_folds=5):
        if self.type == 'allstocks':
            stocks = AllStocksModel.objects.all()
        elif self.type == 'snp500':
            stocks = SNP500Model.objects.all()

        input = []
        for eachStock in stocks:
            stockName = eachStock.stockName
            try:
                stock_data = backend.StockData.Instance().get_historical_stock_data(stockName)
            except Exception as err:
                print "Error getting data for " + stockName + " " + str(err)
                continue
            if len(stock_data) < 1:
                continue
            if stock_data.iloc[-1].close >= 100:
                input.append((stockName, stock_data, self.num_days, self.look_ahead))

        pool = multiprocessing.Pool(20)
        for eachInput in input:
            pool.apply_async(get_feature_label_for_stocks, args=(eachInput,), callback=self.async_callback)
        pool.close()
        pool.join()

        #partition the data into k partitions
        eachPart = self.num_days/k_folds
        parted_data = []
        parted_labels = []
        for i in range(k_folds):
            if i < k_folds-1:
                temp_data = self.results[i*eachPart:(i+1)*eachPart]
                temp_labels = self.labels[i*eachPart:(i+1)*eachPart]
                #parted_data.append(self.results[i*eachPart:(i+1)*eachPart])
                #parted_labels.append(self.labels[i*eachPart:(i+1)*eachPart])
            else:
                temp_data = self.results[i*eachPart:len(self.results)]
                temp_labels = self.labels[i*eachPart:len(self.results)]
                #parted_data.append(self.results[i*eachPart:len(self.results)])
                #parted_labels.append(self.labels[i*eachPart:len(self.results)])
            temp_list_data = []
            temp_list_labels = []
            for p in range(len(temp_data)):
                eachDataRow = temp_data[p]
                eachLabelRow = temp_labels[p]
                for q in range(len(eachDataRow)):
                    temp_list_data.append(eachDataRow[q])
                    temp_list_labels.append(eachLabelRow[q])
            parted_data.append(temp_list_data)
            parted_labels.append(temp_list_labels)

        #return (parted_data, parted_labels)

        all_outcomes = []
        #for C in [ 0.1, 1, 10, 100, 1000]:
        for C in [0.01]:
            all_inputs = []
            #Split into train and test sets, run classification and verification and return the results
            for i in range(k_folds-1):
                train = []
                train_labels = []
                test = []
                test_labels = []
                for j in range(0,i+1):
                    train = train + parted_data[j]
                    train_labels = train_labels + parted_labels[j]
                for j in range(i+1, k_folds):
                    test = test + parted_data[j]
                    test_labels = test_labels + parted_labels[j]
                all_inputs.append((train, train_labels, test, test_labels, C))

            # self.cointoss_results = []
            # pool = multiprocessing.Pool(5)
            # for eachInput in all_inputs:
            #     pool.apply_async(run_cointoss, args=(eachInput,), callback=self.cointoss_async_callback)
            # pool.close()
            # pool.join()
            # all_outcomes.append({'Cointoss': self.cointoss_results})

            self.svm_result = []
            pool = multiprocessing.Pool(5)
            for eachInput in all_inputs:
                pool.apply_async(run_libSVMGPU, args=(eachInput,), callback=self.svm_async_callback)
            pool.close()
            pool.join()
            all_outcomes.append({C: self.svm_result})
        return all_outcomes

    def svm_async_callback(self, result):
        self.svm_result.append(result)
    def cointoss_async_callback(self, result):
        self.cointoss_results.append(result)

def run_libSVMGPU((train, train_labels, test, test_labels, C)):
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


    proc = subprocess.Popen(['libsvm_withgpu/svm-train-gpu', '-c', str(C), filepath, output_filepath],
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

        os.remove(filepath)
        os.remove(output_filepath)
    except Exception as e:
        print(str(e))
        return(-1)

    print(accuracy)
    if result_labels[0] > 0.1:
        return(accuracy[0])
    else:
        return(-1)


def run_libSVM((train, train_labels, test, test_labels, C)):
    svm_prob = svm_problem(train_labels, train, isKernel=True)
    param = svm_parameter('-t 0 -c %d -b 0 -h 0' % (C))
    model = svm_train(svm_prob, param)

    import datetime
    model_name = str(len(train))+'-'+str(datetime.datetime.now())
    svm_save_model(model_name, model)

    result_labels, accuracy, values = svm_predict(test_labels, test, model)

    print(accuracy)
    return(accuracy[0])

def run_linearSVM((train_data, train_labels, test_data, test_labels, C)):
    svm_prob = problem(train_labels, train_data)
    param = parameter('-s 2 -c %d -n 8' % (C))
    model = train(svm_prob, param)

    import datetime
    model_name = str(len(train_data))+'-'+str(datetime.datetime.now())

    result_labels, accuracy, values = predict(test_labels, test_data, model)

    successes = 0.0
    fails = 0.0
    for x in range(len(test_labels)):
        if result_labels[x] > 0.1:
            if test_labels[x] > 0.1:
                successes = successes + 1
            else:
                fails = fails + 1
    pos_accuracy = (successes*100.0)/(successes+fails)

    print(str(result_labels))
    print(accuracy)
    print("Positive Accuracy : " +str(pos_accuracy) + " ("+str(successes)+"/"+str(fails)+")")
    return(pos_accuracy)



def run_SVC((train, train_labels, test, test_labels, C)):
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
    print(" C Success/Failure : "+str(C)+" : "+str(iter_success)+" / "+str(iter_fails)+" , Missed "+str(missed)+' , Total '+str(len(test_labels)) +
    ' Percent classified as 1 :'+str((float(iter_success+iter_fails)/len(test_labels))*100) )
    result = iter_outcome
    return result

def run_cointoss((train, train_labels, test, test_labels, C)):
    iter_results = []
    for i in range(len(test_labels)):
        if random.randint(1,2) == 1:
            iter_results.append(1)
        else:
            iter_results.append(0)
    iter_success = 0
    iter_fails = 0
    for x in range(len(test_labels)):
        if iter_results[x] == test_labels[x]:
            iter_success = iter_success + 1
        else:
            iter_fails = iter_fails + 1
    iter_outcome = ((iter_success)/(iter_success*1.0 + iter_fails*1.0))*100

    print(" Cointoss Accuracy =  : "+str(iter_outcome)+'% ('+str(iter_success)+" / "+str(iter_fails)+')')
    result = iter_outcome
    return result



def get_feature_label_for_stocks((stock_name, data, num_days, look_ahead)):
    #import pdb; pdb.set_trace()
    # data = data[:-200]
    # snp_data = snp_data[:-200]
    # nasdaq_data = nasdaq_data[:-200]
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

        index_mom = talib.MOM(index_data, timeperiod=5)
        index_disparity5 = index_data/talib.MA(index_data, timeperiod=5)
        index_macd, index_macd_signal, index_macd_hist = talib.MACD(index_data)
        index_disparity5 = index_disparity5/np.nanmax(np.abs(index_disparity5))

        #nasdaq_mom1 = talib.MOM(nasdaq_data)

        mom10 = mom10/np.nanmax(np.abs(mom10))
        mom3 = mom3/np.nanmax(np.abs(mom3))
        willr10 = willr10/np.nanmax(np.abs(willr10))
        rsi16 = rsi16/np.nanmax(np.abs(rsi16))
        cci12 = cci12/np.nanmax(np.abs(cci12))
        rocr3 = rocr3/np.nanmax(np.abs(rocr3))
        macdhist = macdhist/np.nanmax(np.abs(macdhist))
        natr = natr/np.nanmax(np.abs(natr))
        index_mom = index_mom/np.nanmax(np.abs(index_mom))
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

        feature_matrix = np.column_stack((mom10, mom3, rsi16, cci12, macdhist, ado, willr10, disparity5, disparity10, beta, index_macd_hist, labels))
        #feature_matrix = np.column_stack((mom10, mom3, rsi16, cci12, percentk, percentd, willr10, disparity5, disparity10, index_disparity5, labels))
        # feature_matrix = np.column_stack((mom10, mom3, rsi16, cci12, macdhist, percentk, percentd, ado,
        # willr10, disparity5, disparity10, index_mom,
        # shift(disparity10, 1, cval=np.NaN), shift(disparity10, 2, cval=np.NaN), shift(disparity10, 3, cval=np.NaN),
        #  labels))

        #feature_matrix = np.column_stack((mom10, mom3, rsi16, cci12, macdhist, percentk, percentd, ado, willr10, disparity5, disparity10, labels))
        #print("--- %s seconds ---" % (time.time() - start_time))
        feature_matrix = feature_matrix[offset:-(look_ahead)]
    except Exception as e:
        print str(e)
        feature_matrix = np.array([])
    return (stock_name, feature_matrix)



def get_feature_label_for_stocks_rdp((stock_name, data, num_days, look_ahead)):
    #import pdb; pdb.set_trace()
    # data = data[:-200]
    # snp_data = snp_data[:-200]
    # nasdaq_data = nasdaq_data[:-200]
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
        print str(e)
        feature_matrix = np.array([])
    return (stock_name, feature_matrix)


def get_label(data, look_ahead):
    #data - numpy array
    labels = np.array([])
    data_len = data.size
    for i in range(data_len):
        if i+look_ahead >= data_len:
            labels = np.append(labels, np.nan)
            continue
        sma = 0
        for s in range(1,look_ahead+1):
            sma = sma + data[i+s]
        sma = sma/look_ahead
        if sma > data[i]:
            labels = np.append(labels, 1)
        else:
            labels = np.append(labels, -1)

    return labels

# def get_label(data, look_ahead):
#     #data - numpy array
#     cost = 10
#     tax = 30
#     money = 1000
#     labels = np.array([])
#     data_len = data.size
#     for i in range(data_len):
#         if i+look_ahead >= data_len:
#             labels = np.append(labels, np.nan)
#             continue
#         buyPrice = data[i]
#         ad = 0
#         found = False
#         num_stocks = money/buyPrice
#         for x in range(look_ahead):
#             ad = ad + 1
#             current_data = data[i+ad]
#             currentPrice = current_data
#             if currentPrice > buyPrice:
#                 diff = (currentPrice-buyPrice)*num_stocks
#                 profit = diff - (0.3*diff)
#                 if profit >= cost:
#                     found = True
#                     break
#         if found:
#             labels = np.append(labels, 1)
#         else:
#             labels = np.append(labels, -1)
#
#     return labels

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
