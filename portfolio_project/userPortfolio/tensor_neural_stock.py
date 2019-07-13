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


import tensorflow as tf


class EnsambleClassifier:
    def __init__(self, date='', type='snp500', num_days=500):
        self.date = date
        self.type = type
        self.look_ahead = 10
        self.min_price = 100
        self.gamma = 1.3
        self.C = 500
        self.epsilon = 0.001
        self.num_days = num_days
        self.results = []
        self.labels = []
        self.period_features_lock = threading.Lock()
        self.model_data_loc = "tensor_model_data/"
        self.progress_file = "model_progress"
        self.tensorflow_model_file = "large_model.ckpt"

    def build_classifier(self):
        pass


    def async_callback_mlp_features(self, result):
        if result == '':
            return
        stock_name, data = result
        #print stock_name
        # if data.shape[0] < self.num_days:
        #     print "Error in stock data: "+stock_name
        #     return

        features = []
        labels = []
        for i in range(len(data)):
            if np.isnan(data[i][-1]):
                labels.append(np.nan)

            else:
                labels.append(int(data[i][-1]))

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
            #ret = get_feature_label_for_stocks_raw(eachStockData)
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
                    combined[fk] = (train_data, train_labels, test_data, test_labels, 5, 500, 10, eachFeature[fk]['stockName'], None)
            period_train_test_data.append(combined)

        self.combined_result = []
        all_mlp_input = [x['mlp'] for x in period_train_test_data]
        pool = multiprocessing.Pool(1)

        for eachTrainTest in period_train_test_data:
            #pool.apply_async(run_MLP, args=(eachTrainTest['mlp'],), callback=self.mlp_prediction_callback)
            self.mlp_prediction_callback(run_MLP(eachTrainTest['mlp']))
        pool.close()
        pool.join()

        picked_stocks = [x['mlp_ret']['stockName'] for x in self.combined_result if x['mlp_ret']['prediction'][0] == 9]
        return picked_stocks


    def train_tensor_model(self, num_train_days, num_test_days, stockNames=[]):
        num_classes = 6
        #verify that the model data loc is present and create it
        if not os.path.exists(self.model_data_loc):
            os.makedirs(self.model_data_loc)

        #train for number of days and test for test_period
        offset = 150
        slice = (2*self.look_ahead) + num_test_days + num_train_days + offset
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

        checkpoint_file_path = self.model_data_loc + self.tensorflow_model_file
        progress_file_path = self.model_data_loc + self.progress_file
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

            #check if the number of trainig days is at least 20 after offset and test days and look ahead
            if len(stock_data) > self.look_ahead + num_test_days + 20 + offset:
                stock_slice = stock_data[-start_ind:-end_ind]
                if stock_slice.iloc[-1].close > self.min_price:
                    period_stock_data.append((stockName, stock_slice, self.num_days+self.look_ahead, self.look_ahead, offset, True, num_classes))

        pool = multiprocessing.Pool(16)
        for eachStockData in period_stock_data:
            ret = get_feature_label_for_stocks_raw(eachStockData)
            pool.apply_async(get_feature_label_for_stocks_raw, args=(eachStockData,), callback=self.async_callback_mlp_features)

        pool.close()
        pool.join()
        period_train_test_data = []
        pf_keys = self.period_features.keys()

        combined_test_features = []
        combined_test_labels = []
        combined_train_features = []
        combined_train_labels = []
        stock_train_data = []
        stock_keys = self.period_features.keys()
        for each_stock_key in stock_keys:
            features = self.period_features[each_stock_key]['mlp']['features']
            stock_name = self.period_features[each_stock_key]['mlp']['stockName']
            labels = self.period_features[each_stock_key]['mlp']['labels']

            train_start_ind = 0
            train_end_ind = -( (2*self.look_ahead) + num_test_days - 1)
            test_start_ind = -(self.look_ahead + num_test_days)
            test_end_ind = -(self.look_ahead)

            train_data = features[ : train_end_ind]
            train_labels = labels[ : train_end_ind]
            test_data = features[test_start_ind : test_end_ind]
            test_labels = labels[test_start_ind : test_end_ind]

            combined_test_features.extend(test_data)
            combined_test_labels.extend(test_labels)
            combined_train_features.extend(train_data)
            combined_train_labels.extend(train_labels)

            stock_train_dic = {'stockName':stock_name, 'trainData':train_data,
                'trainLabels':train_labels}
            stock_train_data.append(stock_train_dic)

        network_depth = 5
        network_width = 500

        checkpoint_file_path = self.model_data_loc + self.tensorflow_model_file
        progress_file_path = self.model_data_loc + self.progress_file
        try:
            progress = cPickle.load(open(progress_file_path, 'rb'))
        except Exception as ex:
            print(str(ex))
            progress = []
        test_accuracies = []
        # for each_train_set in stock_train_data:
        #     current_stock_name = each_train_set['stockName']
        #     if current_stock_name in progress:
        #         print("Model trainig done for " + current_stock_name)
        #         continue


            # result = run_MLP((each_train_set['trainData'], each_train_set['trainLabels'], combined_test_features,
            #     combined_test_labels, network_depth, network_width, num_classes,
            #     each_train_set['stockName'], checkpoint_file_path))
        current_stock_name = "All_Stocks"
        result = run_MLP((combined_train_features, combined_train_labels, combined_test_features,
            combined_test_labels, network_depth, network_width, num_classes,
            current_stock_name, checkpoint_file_path))
        progress.append(current_stock_name)
        cPickle.dump(progress, open(progress_file_path, 'wb'))

        print("Finished training for "+current_stock_name)
        print("Current Test Accuracy : "+str(result['accuracy']))
        test_accuracies.append(result['accuracy'])

        class_success = {}
        class_fail = {}
        for i in range(num_classes):
            class_success[i] = 0
            class_fail[i] = 0
        predictions = result['prediction']
        for i in range(len(predictions)):
            if predictions[i] == combined_test_labels[i]:
                class_success[predictions[i]] += 1
            else:
                class_fail[predictions[i]] += 1

        class_accuracies = {}
        for ec in class_success.keys():
            acc = (class_success[ec]*100.0)/(class_success[ec]+class_fail[ec])
            class_accuracies[ec] = acc
        import pdb; pdb.set_trace()
        return class_accuracies





    def mlp_prediction_callback(self, mlp_ret):
        test_labels = mlp_ret['prediction']
        poss = []
        mlp_labels = mlp_ret['prediction']
        for i in range(len(test_labels)):
            #temp_dic = {'svm_ret':svm_ret, 'svm_negative_ret':svm_negative_ret, 'svm_negative_ret2':svm_negative_ret2}
            temp_dic = {'mlp_ret':mlp_ret}
            temp_dic['actual'] = test_labels[i]
            self.combined_result.append(temp_dic)
            print "Processed " + str(len(self.combined_result)) + " / " + str(len(self.period_features))

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



def get_feature_label_for_stocks_raw((stock_name, data, num_days, look_ahead, offset, blind, num_classes)):
    start_time = time.time()
    # offset = 40
    slice_len = num_days+look_ahead+offset
    data_frame = data.tail(slice_len)
    index_data = data_frame['close_index'].values
    close = data_frame['close'].values
    high = data_frame['high'].values
    low = data_frame['low'].values
    volume = data_frame['volume'].values
    close_raw = data_frame['close'].values

    if True:
    #try:
        close -= np.mean(close)
        close /= np.absolute(close).max()
        #close = np.clip(close, -0.99, 0.99)

        high -= np.mean(high)
        high /= np.absolute(high).max()
        #high = np.clip(high, -0.99, 0.99)

        low -= np.mean(low)
        low /= np.absolute(low).max()
        #low = np.clip(low, -0.99, 0.99)

        volume -= np.mean(volume)
        volume /= np.absolute(volume).max()
        #volume = np.clip(volume, -0.99, 0.99)

        index_data -= np.mean(index_data)
        index_data /= np.absolute(index_data).max()
        #index_data = np.clip(index_data, -0.99, 0.99)

        labels = get_label(close_raw, look_ahead, num_classes)


        #feature_matrix = np.column_stack((mom10, mom3, rsi16, cci12, macdhist, ado, willr10, disparity5, disparity10, beta, index_macd_hist, labels))
        num_feature_days = 20
        num_index_days = 20

        #feature_matrix = np.column_stack((close))

        stack = []

        for i in range(num_feature_days):
            #feature_matrix = np.concatenate((feature_matrix, shift(close, i, cval=np.NaN)), axis=1)
            stack.append(shift(close, i, cval=np.NaN))

        for i in range(num_feature_days):
            #feature_matrix = np.concatenate((feature_matrix, shift(high, i, cval=np.NaN)), axis=1)
            stack.append(shift(high, i, cval=np.NaN))

        for i in range(num_feature_days):
            #feature_matrix = np.concatenate((feature_matrix, shift(low, i, cval=np.NaN)), axis=1)
            stack.append(shift(low, i, cval=np.NaN))

        for i in range(num_feature_days):
            #feature_matrix = np.concatenate((feature_matrix, shift(volume, i, cval=np.NaN)), axis=1)
            stack.append(shift(volume, i, cval=np.NaN))

        for i in range(num_index_days):
            #feature_matrix = np.concatenate((feature_matrix, shift(index_data, i, cval=np.NaN)), axis=1)
            stack.append(shift(index_data, i, cval=np.NaN))

        stack.append(labels)

        feature_matrix = np.column_stack(stack,)

        # feature_matrix = np.column_stack((close, high, low, volume,
        # shift(close, 1, cval=np.NaN), shift(high, 1, cval=np.NaN), shift(low, 1, cval=np.NaN),shift(volume, 1, cval=np.NaN),
        # shift(close, 2, cval=np.NaN), shift(high, 2, cval=np.NaN), shift(low, 2, cval=np.NaN),shift(volume, 2, cval=np.NaN),
        # shift(close, 3, cval=np.NaN), shift(high, 3, cval=np.NaN), shift(low, 3, cval=np.NaN),shift(volume, 3, cval=np.NaN),
        # shift(close, 4, cval=np.NaN), shift(high, 4, cval=np.NaN), shift(low, 4, cval=np.NaN),shift(volume, 4, cval=np.NaN),
        # shift(close, 5, cval=np.NaN), shift(high, 5, cval=np.NaN), shift(low, 5, cval=np.NaN),shift(volume, 5, cval=np.NaN),
        # shift(close, 6, cval=np.NaN), shift(high, 6, cval=np.NaN), shift(low, 6, cval=np.NaN),shift(volume, 6, cval=np.NaN),
        # shift(close, 7, cval=np.NaN), shift(high, 7, cval=np.NaN), shift(low, 7, cval=np.NaN),shift(volume, 7, cval=np.NaN),
        # shift(close, 8, cval=np.NaN), shift(high, 8, cval=np.NaN), shift(low, 8, cval=np.NaN),shift(volume, 8, cval=np.NaN),
        # shift(close, 9, cval=np.NaN), shift(high, 9, cval=np.NaN), shift(low, 9, cval=np.NaN),shift(volume, 9, cval=np.NaN),
        # shift(close, 10, cval=np.NaN), shift(high, 10, cval=np.NaN), shift(low, 10, cval=np.NaN),shift(volume, 10, cval=np.NaN),
        # index_data,
        # shift(index_data, 1, cval=np.NaN), shift(index_data, 2, cval=np.NaN), shift(index_data, 3, cval=np.NaN),
        # shift(index_data, 4, cval=np.NaN), shift(index_data, 5, cval=np.NaN),
        # labels))

        if not blind:
            feature_matrix = feature_matrix[offset:-(look_ahead)]
        else:
            feature_matrix = feature_matrix[offset:]
    # except Exception as e:
    #     print ("portmlp: get_feature_label_for_stocks_raw : " + str(e))
    #     feature_matrix = np.array([])
    return (stock_name, feature_matrix)



def get_label(data, look_ahead, num_classes):
    num_bins = num_classes
    bin_range = [-10.0, 10.0]

    bin_sz = (bin_range[1]-bin_range[0]) / (num_bins-2)
    ranges = []
    ranges.append((-sys.maxint-1, bin_range[0]))
    for i in range(0,num_bins-2):
        ranges.append((bin_range[0]+(i*bin_sz), bin_range[0]+((i+1)*bin_sz)))

    ranges.append((bin_range[1], sys.maxint))


    #data - numpy array
    labels = np.array([])
    data_len = data.size
    for i in range(data_len):
        if i+look_ahead >= data_len:
            labels = np.append(labels, np.nan)
            continue
        change = ((data[i+look_ahead] - data[i])/data[i])*100.0
        for c in range(len(ranges)):
            if  ranges[c][0] <= change <= ranges[c][1]:
                labels = np.append(labels, c)
                break
    return labels

def build_tensorflow_graph(feature_len, num_classes, width, depth, dropout_prob, learning_rate):
    x = tf.placeholder("float", [None, feature_len], name="x")
    y = tf.placeholder("float", [None, num_classes], name="y")

    #tf variables to store weights and baises
    weights = []
    biases = []

    all_layers = [x]
    layer_widths = [feature_len]
    for i in range(1,depth+1):
        W = tf.Variable(tf.random_normal([layer_widths[i-1], width]))
        B = tf.Variable(tf.random_normal([width]))
        weights.append(W)
        biases.append(B)
        layer_widths.append(width)

        layer = tf.add(tf.matmul(all_layers[i-1], W), B)
        layer = tf.nn.relu(layer)
        layer = tf.nn.dropout(layer, dropout_prob)
        all_layers.append(layer)

    W = tf.Variable(tf.random_normal([layer_widths[len(layer_widths)-1], num_classes]))
    B = tf.Variable(tf.random_normal([num_classes]))
    weights.append(W)
    biases.append(B)
    out_layer_mul = tf.matmul(all_layers[len(all_layers)-1], W)
    out_layer = tf.add(out_layer_mul, B, name="out_layer")

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=y), name="cost")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="optimizer").minimize(cost)
    import pdb; pdb.set_trace()
    reg_constant = 5  # Choose an appropriate one.
    for i in range(len(weights)):
        cost = cost + (reg_constant*tf.nn.l2_loss(weights[i])) + \
            (reg_constant*tf.nn.l2_loss(biases[i]))

    return (out_layer, optimizer, cost, x, y)


def run_MLP((train, train_labels, test, test_labels, depth, width, num_classes, stockName, checkpoint_file_path)):
    learning_rate = 0.00001
    training_epochs = 1000
    batch_size = 10
    display_step = 10
    dropout_prob = 0.75

    tf.reset_default_graph()

    X_train = np.array(train, dtype=np.float32)
    #y_train = np.array(train_labels, dtype=np.int32)
    X_test = np.array(test, dtype=np.float32)

    renn = RandomUnderSampler()
    features_res, labels_res =renn.fit_sample(X_train, np.array(train_labels, dtype=np.int32))

    X_train = features_res

    feature_len = (X_train[0].shape)[0]

    #oversample the imbalanced classes using smote

    y_train = np.zeros((X_train.shape[0], num_classes))
    y_train[np.arange(len(labels_res)), labels_res] = 1

    y_test = np.zeros((X_test.shape[0], num_classes))
    y_test[np.arange(len(test_labels)), test_labels] = 1

    # if not tf.train.checkpoint_exists(checkpoint_file_path):

    rnd_indices = np.random.rand(len(X_train)) < 1.0
    X_train = X_train[rnd_indices]
    y_train = y_train[rnd_indices]



    final_train_accuracy = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    #Initialize the tf variables
    init = tf.global_variables_initializer()

    #Lauch the graph
    with tf.Session(config=config) as sess:

        #check if the chekpoint file exists and restore the models
        #if checkpoint_file_path and os.path.isfile(checkpoint_file_path):
        if tf.train.checkpoint_exists(checkpoint_file_path) and False:
            saver = tf.train.import_meta_graph(checkpoint_file_path+".meta")
            saver.restore(sess, checkpoint_file_path)
            g = tf.get_default_graph()
            out_layer = g.get_tensor_by_name("out_layer:0")
            optimizer = g.get_operation_by_name("optimizer")
            cost = g.get_tensor_by_name("cost:0")
            x = g.get_tensor_by_name("x:0")
            y = g.get_tensor_by_name("y:0")
            print("Restored model from file " + checkpoint_file_path)
        else:
            (out_layer, optimizer, cost, x, y) = build_tensorflow_graph(feature_len, num_classes, width, depth, dropout_prob, learning_rate)
            saver = tf.train.Saver()

        #Initialize the tf variables
        init = tf.global_variables_initializer()
        sess.run(init)
        #Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int( (X_train.shape)[0] / batch_size)

            for i in range(total_batch):
                excerpt = slice((i*batch_size), ((i+1)*batch_size))
                batch_x = X_train[excerpt]
                batch_y = y_train[excerpt]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch

            # Display logs per epoch step
            if epoch % display_step == 0:
                if(X_test.shape[0] > 0):
                    #get predictions by running on test inputs
                    final_predictions = sess.run(tf.argmax(out_layer, 1), feed_dict={x:X_test})

                    correct_prediction = tf.equal(tf.argmax(out_layer, 1), tf.argmax(y, 1))
                    # Calculate accuracy
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    test_accuracy = accuracy.eval({x:X_test, y:y_test})
                else:
                    final_predictions = []
                    test_accuracy = 0
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                   "{:.9f}".format(avg_cost), "Test accuracy=", "{:.9f}".format(test_accuracy))


                confusion_mat = tf.contrib.metrics.confusion_matrix(test_labels, final_predictions, num_classes=num_classes)
                print confusion_mat.eval()
            final_train_accuracy = avg_cost

        print("Optimization Finished!")

        #is checkpoint_file_path given.. save the model
        if checkpoint_file_path:
            save_path = saver.save(sess, checkpoint_file_path)
            print("Model saved in file: %s" % save_path)

        if(X_test.shape[0] > 0):
            #get predictions by running on test inputs
            final_predictions = sess.run(tf.argmax(out_layer, 1), feed_dict={x:X_test})

            correct_prediction = tf.equal(tf.argmax(out_layer, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            test_accuracy = accuracy.eval({x:X_test, y:y_test})
        else:
            final_predictions = []
            test_accuracy = 0


    result = {'stockName':stockName, 'accuracy': test_accuracy,
    'training_accuracy' : final_train_accuracy, 'prediction' : final_predictions }

    return result

def test_model((train, train_labels, test, test_labels, depth, width, num_classes, stockName, checkpoint_file_path)):
    learning_rate = 0.001
    training_epochs = 1000
    batch_size = 1000
    display_step = 1
    dropout_prob = 0.75

    tf.reset_default_graph()

    X_train = np.array(train, dtype=np.float32)
    #y_train = np.array(train_labels, dtype=np.int32)
    X_test = np.array(test, dtype=np.float32)

    feature_len = (X_train[0].shape)[0]

    y_train = np.zeros((X_train.shape[0], num_classes))
    for i in range(X_train.shape[0]):
        y_train[i][train_labels[i]] = 1.0

    y_test = np.zeros((X_test.shape[0], num_classes))
    for i in range(X_test.shape[0]):
        y_test[i][test_labels[i]] = 1.0

    # if not tf.train.checkpoint_exists(checkpoint_file_path):
    x = tf.placeholder("float", [None, feature_len])
    y = tf.placeholder("float", [None, num_classes])

    #tf variables to store weights and baises
    weights = []
    biases = []

    all_layers = [x]
    layer_widths = [feature_len]
    for i in range(1,depth+1):
        W = tf.Variable(tf.random_normal([layer_widths[i-1], width]))
        B = tf.Variable(tf.random_normal([width]))
        weights.append(W)
        biases.append(B)
        layer_widths.append(width)

        layer = tf.add(tf.matmul(all_layers[i-1], W), B)
        layer = tf.nn.relu(layer)
        layer = tf.nn.dropout(layer, dropout_prob)
        all_layers.append(layer)

    W = tf.Variable(tf.random_normal([layer_widths[len(layer_widths)-1], num_classes]))
    B = tf.Variable(tf.random_normal([num_classes]))
    weights.append(W)
    biases.append(B)
    out_layer = tf.matmul(all_layers[len(all_layers)-1], W) + B

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    #Initialize the tf variables
    init = tf.global_variables_initializer()

    final_train_accuracy = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:

        #check if the chekpoint file exists and restore the models
        #if checkpoint_file_path and os.path.isfile(checkpoint_file_path):
        if tf.train.checkpoint_exists(checkpoint_file_path):
            saver.restore(sess, checkpoint_file_path)
            print("Restored model from file " + checkpoint_file_path)

        sess.run(init)

        if(X_test.shape[0] > 0):
            #get predictions by running on test inputs
            final_predictions = sess.run(tf.argmax(out_layer, 1), feed_dict={x:X_test})

            correct_prediction = tf.equal(tf.argmax(out_layer, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            test_accuracy = accuracy.eval({x:X_test, y:y_test})
        else:
            final_predictions = []
            test_accuracy = 0


    result = {'stockName':stockName, 'accuracy': test_accuracy,
    'training_accuracy' : final_train_accuracy, 'prediction' : final_predictions }

    return result
#

def plot_confusion_matrix(fig, ax, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.title(title)
    ax.colorbar()
    tick_marks = np.arange(len(classes))
    ax.xticks(tick_marks, classes, rotation=45)
    ax.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    ax.tight_layout()
    ax.ylabel('True label')
    ax.xlabel('Predicted label')
    fig.canvas.flush_events()
    time.sleep(1)
