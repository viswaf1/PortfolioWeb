import datetime, sys, os, multiprocessing, time, pickle, json
import threading
from userPortfolio.models import UserTransactionsModel, UserPortfolioModel, \
AllStocksModel, SNP500Model, USDForexModel, ModelFileData
from django.conf import settings
import csv, hashlib, itertools
import pandas, urllib.request, urllib.error, urllib.parse, csv, random, datetime, string, subprocess
from pytz import timezone
from dateutil.relativedelta import relativedelta
from math import pi
import string, random
import userPortfolio.backend as backend
import numpy as np
from operator import itemgetter
import talib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from time import sleep
from celery import shared_task
from celery.result import allow_join_result

from tensorflow.keras import backend as K

def single_class_accuracy(interesting_class_id):
    def fn(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # Replace class_id_preds with class_id_true for recall here
        accuracy_mask = K.cast(K.equal(class_id_preds, interesting_class_id), 'int32')
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc
    return fn



class TensorDNNModelBuilder:

    def __init__(self, modelParams):
        self.modelParmas = modelParams
        self.lock = multiprocessing.Lock()


    def BeginNewBuild(self):
        print("Starting New Model")
        self.builderRunning = True
        self.featureLabelTasksIds = []
        # hbThread = threading.Thread(target=self.HeartbeatThread)
        # hbThread.start()

        modelEntry = ModelFileData.objects.get(id=self.modelParmas['modelId'])
        # modelEntry.save()
        self.modelPath = modelEntry.path

        if not os.path.exists(self.modelPath):
            os.mkdir(self.modelPath)
            print("Directory " , self.modelPath ,  " Created ")
        else:
            print("Directory " , self.modelPath ,  " already exists")

        self.allFeatures = []
        self.allLabels = []

        self.BuildTrainTestData()
        # self.SplitTrainValidation()
        self.WriteFeatureStats()
        modelEntry = ModelFileData.objects.get(id=self.modelParmas['modelId'])
        modelEntry.state = 'NETWORKTRAIN'
        modelEntry.save()

        self.TrainTensorFlowModel()

        self.builderRunning = False
        # hbThread.join()


    def HeartbeatThread(self):
        while(self.builderRunning):
            file = open(settings.MODELS_STATE_FILE, 'w')
            timestr = datetime.datetime.now().strftime(settings.DATETIMESTRFORMAT)
            file.write(timestr)
            print("HeartBeat!!!!")
            sleep(10)

    def FeatureLabelAsyncCallback(self, result):
        self.lock.acquire()
        self.allFeatures.extend(result[0])
        self.allLabels.extend(result[1])
        self.lock.release()

    def BuildTrainTestData(self):

        self.validationFeatures = []
        self.validationLabels = []
        self.trainingFeatures = []
        self.trainingLabels = []

        modelEntry = ModelFileData.objects.get(id=self.modelParmas['modelId'])
        modelEntry.state = 'FEATUREGEN'
        modelEntry.save()

        backend.StockData.Instance().offline = False
        if self.modelParmas['indexName'] == 'allstocks':
                stocks = AllStocksModel.objects.all()
        elif self.modelParmas['indexName'] == 'SNP500':
            stocks = SNP500Model.objects.all()
        elif self.modelParmas['indexName'] == 'forex':
            stocks = USDForexModel.objects.all()
        stockNames = []
        for eachStock in stocks:
            stockNames.append(eachStock.stockName)

        # pool = multiprocessing.Pool(16)

        for stockName in stockNames:
            try:
                stock_data = backend.StockData.Instance().get_historical_stock_data(stockName)
            except Exception as err:
                print("Error getting data for " + stockName + " " + str(err))
                continue
            if len(stock_data) < 1:
                continue

            startInd = min(self.modelParmas['trainingDays'], len(stock_data.index))
            endInd = 1
            stockSlice = stock_data[-startInd:-endInd]
            stockSliceJson  = stockSlice.to_json()

            currTaskId = GetDataFeatures.delay(stockSliceJson, self.modelParmas['featureLength'],
                                self.modelParmas['positiveLabelValue'],
                                self.modelParmas['holdDays'])
            self.featureLabelTasksIds.append(currTaskId)
        resultCount = 0
        for eachTask in self.featureLabelTasksIds:
            while not eachTask.ready():
                sleep(1)
            with allow_join_result():
                (featuresNpSer, labels, success) = eachTask.get()

            if not success:
                continue
            featuresNp = np.array(featuresNpSer)
            featuresList = []
            numValid = int((featuresNp.shape[0] * self.modelParmas['validationPercent'])/100.0)
            numTrain = featuresNp.shape[0] - (numValid + self.modelParmas['featureLength'])
            for r in range(numTrain):
                self.trainingFeatures.append(featuresNp[r])
                self.trainingLabels.append(labels[r])
            for r in range(numTrain + self.modelParmas['featureLength'],
                            numTrain + self.modelParmas['featureLength'] + numValid):
                self.validationFeatures.append(featuresNp[r])
                self.validationLabels.append(labels[r])


        featuresNp = np.asarray(self.trainingFeatures)
        colMean = featuresNp.mean(axis=0)
        colStd = featuresNp.std(axis=0)

        for ind in range(len(self.trainingFeatures)):
            self.trainingFeatures[ind] = np.clip((self.trainingFeatures[ind] - colMean) / colStd, -2, 2)

        for ind in range(len(self.validationFeatures)):
            self.validationFeatures[ind] = np.clip((self.validationFeatures[ind] - colMean) / colStd, -2, 2)


        # from celery.contrib import rdb
        # rdb.set_trace()

    def SplitTrainValidation(self):

        numValidation = int((len(self.allFeatures) * self.modelParmas['validationPercent'])/100.0)
        samples = np.random.choice(len(self.allFeatures), numValidation)

        self.validationFeatures = []
        self.validationLabels = []
        self.trainingFeatures = []
        self.trainingLabels = []

        deleteInds = []
        for eachSample in samples:
            self.validationFeatures.append(self.allFeatures[eachSample])
            self.validationLabels.append(self.allLabels[eachSample])
            deleteInds.append(eachSample)

        deleteInds.sort()
        dInd = 0
        for fInd in range(len(self.allFeatures)):
            if fInd == deleteInds[dInd]:
                dInd += 1
            else:
                self.trainingFeatures.append(self.allFeatures[fInd])
                self.trainingLabels.append(self.allLabels[fInd])


    def WriteFeatureStats(self):
        stats = {}
        stats['NumFeatures'] = len(self.trainingFeatures)
        stats['LabelCounts'] = {}
        for eachLabel in self.trainingLabels:
            labC = stats['LabelCounts'].get(eachLabel, 0)
            stats['LabelCounts'][eachLabel] = labC + 1

        validStats = {}
        validStats['NumFeatures'] = len(self.validationFeatures)
        validStats['LabelCounts'] = {}
        for eachLabel in self.validationLabels:
            labC = validStats['LabelCounts'].get(eachLabel, 0)
            validStats['LabelCounts'][eachLabel] = labC + 1

        # from celery.contrib import rdb
        # rdb.set_trace()
        fstatFilePath = os.path.join(self.modelPath, settings.FEATURE_STAT_FILENAME)
        pickle.dump(stats, open(fstatFilePath, 'wb'))

        validFstatFilePath = os.path.join(self.modelPath, settings.FEATURE_VALIDATION_STAT_FILENAME)
        pickle.dump(validStats, open(validFstatFilePath, 'wb'))

    def TrainTensorFlowModel(self):
        modelDir = os.path.join(self.modelPath, "TensorModel")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=modelDir)

        trainNp = np.asarray(self.trainingFeatures)

        trainLabelsLogits = []
        for eachLabel in self.trainingLabels:
            temp = [0,0,0]
            temp[eachLabel] = 1
            trainLabelsLogits.append(temp)
        trainLabelsNp = np.asarray(trainLabelsLogits)

        validNp = np.asarray(self.validationFeatures)
        validLabelsLogits = []
        for eachLabel in self.validationLabels:
            temp = [0,0,0]
            temp[eachLabel] = 1
            validLabelsLogits.append(temp)
        validLabelsNp = np.asarray(validLabelsLogits)

        model = tf.keras.Sequential([
            keras.layers.BatchNormalization(),
            keras.layers.InputLayer(input_shape=(trainNp.shape[1],) ),
            keras.layers.Dense(512, activation=tf.nn.relu),
            keras.layers.Dropout(0.5),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(512, activation=tf.nn.relu),
            keras.layers.Dropout(0.5),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(512, activation=tf.nn.relu),
            # keras.layers.Dropout(0.5),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(3, activation=tf.nn.softmax)
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
                      metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.FalsePositives(), single_class_accuracy(1)])
              # metrics=['accuracy'])
        
        

        # from celery.contrib import rdb
        # rdb.set_trace()

        # for epoch in range(2000):
        training_history = model.fit(trainNp, trainLabelsNp, epochs=2000000,
                    validation_data = (validNp, validLabelsNp),
                    callbacks=[tensorboard_callback])
            # test_loss, test_acc = model.evaluate(validNp, validLabelsNp)
            # print('Test accuracy:', test_acc)

    def TrainTensorFlowModelEst(self):
        modelDir = os.path.join(self.modelPath, "TensorModel")
        featureVec = tf.feature_column.numeric_column('featureVec', shape=self.trainingFeatures[0].shape[0])
        featureCol = [featureVec]
        trainFeatureTens = np.asarray(self.trainingFeatures)
        trainLabelTens = np.asarray(self.trainingLabels)
        trainDic = {'featureVec':trainFeatureTens}
        validationFeatureTens = np.asarray(self.validationFeatures)
        validationLabelTens = np.asarray(self.validationLabels)
        validDic = {'featureVec':validationFeatureTens}
        batchSize = 5000
        def train_input_fn():
            dataset = tf.data.Dataset.from_tensor_slices((trainDic,
                                                                trainLabelTens))
            return dataset.shuffle(batchSize+500).repeat().batch(batchSize)

        def eval_input_fn():
            dataset = tf.data.Dataset.from_tensor_slices((validDic, validationLabelTens))
            return dataset.shuffle(batchSize+500).repeat().batch(batchSize)

        config = tf.estimator.RunConfig()
        config = config.replace(save_summary_steps=100)
        config = config.replace(save_checkpoints_secs=40)

        # optimizer_adam= tf.train.AdamOptimizer(learning_rate=0.01)
        model=tf.estimator.DNNClassifier(hidden_units=[20,20,20], feature_columns=featureCol,
                                         config=config, model_dir=modelDir,
                                         optimizer=lambda: tf.compat.v1.train.AdamOptimizer(
                                             learning_rate=0.01)
                                         )


        # model=tf.estimator.DNNClassifier(hidden_units=[10,10,10], feature_columns=featureCol,
        #                                  config=config,
        #                                     optimizer=tf.compat.v1.train.ProximalAdagradOptimizer(
        #                                   learning_rate=0.01,
        #                                   l1_regularization_strength=0.01),
        #                                  model_dir=modelDir,
        #                                 dropout=0.1
        #                                  )


        # model.train(input_fn=lambda: train_input_fn(), steps=200000)
        trainSpec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(), max_steps=1000000)
        evalSpec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=1, throttle_secs = 5)

        # trainSpec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(), max_steps=1)
        # evalSpec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=1)

        # for epoch in range(20000):
        tf.estimator.train_and_evaluate(model, trainSpec, evalSpec)

        eval_results = model.evaluate(input_fn=eval_input_fn, steps=1)
        print(eval_results)


@shared_task
def GetDataFeatures(dataSliceJson, featureLen, positiveLabelPercent, holdDays):
    dataSlice = pandas.read_json(dataSliceJson)
    features = []
    labels = []

    openArray = dataSlice['open'].values
    closeArray = dataSlice['close'].values
    highArray = dataSlice['high'].values
    lowArray = dataSlice['low'].values

    # openDiff = (np.clip(((openArray[1:] - openArray[:-1]) / closeArray[:-1]) * 100, -2, 2) + 2) / 4
    # closeDiff = (np.clip(((closeArray[1:] - closeArray[:-1]) / closeArray[:-1]) * 100, -2, 2) + 2) / 4
    # highDiff =  (np.clip(((highArray[1:] - highArray[:-1]) / closeArray[:-1]) * 100, -2, 2) +2) / 4
    # lowDiff = (np.clip(((lowArray[1:] - lowArray[:-1]) / closeArray[:-1]) * 100, -2, 2) + 2) / 4


    openDiff = ((openArray[1:] - openArray[:-1]) / openArray[:-1]) * 100
    closeDiff = ((closeArray[1:] - closeArray[:-1]) / closeArray[:-1]) * 100
    highDiff = ((highArray[1:] - highArray[:-1]) / highArray[:-1]) * 100
    lowDiff = ((lowArray[1:] - lowArray[:-1]) / lowArray[:-1]) * 100



    numFeatures = closeDiff.size  - featureLen - holdDays
    for featureInd in range(0, numFeatures):
        feature = openDiff[featureInd : featureInd+featureLen]
        feature = np.vstack((feature, closeDiff[featureInd : featureInd+featureLen]))
        feature = np.vstack((feature, highDiff[featureInd : featureInd+featureLen]))
        feature = np.vstack((feature, lowDiff[featureInd : featureInd+featureLen]))

        feature = feature.flatten('C')
        features.append(feature)

        # labelOpenValue = highDiff[featureInd+featureLen+holdDays-1]
        gain = ((highArray[1:][featureInd+featureLen+holdDays] - openArray[1:][featureInd+featureLen]) / openArray[1:][featureInd+featureLen])*100
        loss = ((lowArray[1:][featureInd+featureLen+holdDays] - openArray[1:][featureInd+featureLen]) / openArray[1:][featureInd+featureLen])*100
        if loss < -0.5:
            labels.append(2)
        elif gain >= positiveLabelPercent:
            labels.append(1)
        else:
            labels.append(0)
    if len(features) < 1:
        return ([], [], False)

    featuresNp = features[0]
    # labelsNp = labels[0]
    for i in range(1, len(features)):
        featuresNp = np.vstack((featuresNp, features[i]))
        # np.vstack((labelsNp, labels[i]))

    # featuresSer = featuresNp.tobytes()
    # featuresSer = pickle.dumps(featuresNp)
    featuresSer = featuresNp.tolist()
    # labelsSer = labelsNp.tobytes()
    return (featuresSer, labels, True)



        

