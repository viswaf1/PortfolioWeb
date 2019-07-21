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
from time import sleep
from celery import shared_task
from celery.result import allow_join_result



class TensorDNNModelBuilder:

    def __init__(self, modelParams):
        self.modelParmas = modelParams
        self.lock = multiprocessing.Lock()


    def BeginNewBuild(self):
        print("Starting New Model")
        self.builderRunning = True
        self.featureLabelTasksIds = []
        hbThread = threading.Thread(target=self.HeartbeatThread)
        hbThread.start()

        currTime = datetime.datetime.now()
        nameStr = currTime.strftime("%Y-%m-%d-%H-%M-%S")
        randomStr = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(5)])
        modelpath = nameStr+randomStr
        modelpath = os.path.join(settings.MODELS_DIR, modelpath)

        modelEntry = ModelFileData.objects.create(type = "TENSOR_DNN",
                                   path = modelpath, datetime = currTime,
                                   state = "INIT")
        modelEntry.save()
        self.modelPath = modelpath

        if not os.path.exists(self.modelPath):
            os.mkdir(self.modelPath)
            print("Directory " , self.modelPath ,  " Created ")
        else:
            print("Directory " , self.modelPath ,  " already exists")

        self.allFeatures = []
        self.allLabels = []

        self.BuildTrainTestData()


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
            # (features, lables) = self.GetDataFeatures(stockSlice, self.modelParmas['featureLength'],
            #                     self.modelParmas['positiveLabelValue'],
            #                     self.modelParmas['holdDays'])

            # pool.apply_async(GetDataFeatures, args = (stockSlice, self.modelParmas['featureLength'],
            #                     self.modelParmas['positiveLabelValue'],
            #                     self.modelParmas['holdDays'], ), callback = self.FeatureLabelAsyncCallback)
            # self.allFeatures.extend(features)
            # self.allLabels.extend(lables)
            currTaskId = GetDataFeatures.delay(stockSliceJson, self.modelParmas['featureLength'],
                                self.modelParmas['positiveLabelValue'],
                                self.modelParmas['holdDays'])
            self.featureLabelTasksIds.append(currTaskId)
        resultCount = 0
        for eachTask in self.featureLabelTasksIds:
            while not eachTask.ready():
                sleep(1)
            with allow_join_result():
                (featuresNpSer, labels) = eachTask.get()

            featuresNp = np.array(featuresNpSer)
            # labelsNp = np.from_bytes(labelsSer)
            self.allFeatures.append(featuresNp)
            self.allLabels.append(labels)

        #
        # pool.close()
        # pool.join()
        from celery.contrib import rdb
        rdb.set_trace()

@shared_task
def GetDataFeatures(dataSliceJson, featureLen, positiveLabelPercent, holdDays):
    dataSlice = pandas.read_json(dataSliceJson)
    features = []
    labels = []

    openArray = dataSlice['open'].values
    closeArray = dataSlice['close'].values
    highArray = dataSlice['high'].values
    lowArray = dataSlice['low'].values

    openDiff = ((openArray[1:] - openArray[:-1]) / closeArray[:-1]) * 100
    closeDiff = ((closeArray[1:] - closeArray[:-1]) / closeArray[:-1]) * 100
    highDiff =  ((highArray[1:] - highArray[:-1]) / closeArray[:-1]) * 100
    lowDiff = ((lowArray[1:] - lowArray[:-1]) / closeArray[:-1]) * 100


    numFeatures = closeDiff.size  - featureLen - holdDays
    for featureInd in range(0, numFeatures):
        feature = openDiff[featureInd : featureInd+featureLen]
        feature = np.vstack((feature, closeDiff[featureInd : featureInd+featureLen]))
        feature = np.vstack((feature, highDiff[featureInd : featureInd+featureLen]))
        feature = np.vstack((feature, lowDiff[featureInd : featureInd+featureLen]))

        feature = feature.flatten('F')
        features.append(feature)

        labelOpenValue = highDiff[featureInd+featureLen+holdDays-1]
        if labelOpenValue >= 1.0:
            labels.append(1)
        else:
            labels.append(0)

    featuresNp = features[0]
    # labelsNp = labels[0]
    for i in range(1, len(features)):
        featuresNp = np.vstack((featuresNp, features[i]))
        # np.vstack((labelsNp, labels[i]))

    # featuresSer = featuresNp.tobytes()
    # featuresSer = pickle.dumps(featuresNp)
    featuresSer = featuresNp.tolist()
    # labelsSer = labelsNp.tobytes()
    return (featuresSer, labels)


        

