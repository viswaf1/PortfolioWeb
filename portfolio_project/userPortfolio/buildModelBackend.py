import datetime
import sys,os
from django.shortcuts import render, render_to_response
from userPortfolio.models import UserTransactionsModel, UserPortfolioModel, AllStocksModel, SNP500Model, UserProfile
from django.db import IntegrityError
from django.conf import settings
from django.utils import timezone as djangotimezone
import csv, hashlib
from pandas_datareader import data
import pandas, urllib.request, urllib.error, urllib.parse, csv
from pytz import timezone
from dateutil.relativedelta import relativedelta
import datetime
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.resources import CDN
from bokeh.models import Legend, LegendItem
from bokeh.embed import components
from math import pi
from talib.abstract import *
from celery import shared_task
import userPortfolio.TensorDNNModel

def renderModelBuilder(request):

    stockIndices = ['SNP500']
    indexName = stockIndices[0]

    modelParams = {'indexName' : stockIndices[0],
        'trainingDays' : 100,
        'featureLength' : 10,
        'validationPercent' : 10,
        'positiveLabelValue' : 1.0,
        'holdDays' : 1
    }
    context_dict = {}
    context_dict['post_error'] = False
    startingModelRun = False

    errorStr = ''
    if request.method == 'POST':
        indexName = request.POST.get("IndexType")
        # print("Index name is " + indexName)
        if indexName not in stockIndices:
            errorStr += "Invalid Stock Index Type\n"
            context_dict['post_error'] = True
            indexName = stockIndices[0]


        try:
            numTrainingDays = int(request.POST.get("trainingDays"))
        except:
            numTrainingDays = -1
        if numTrainingDays < 10:
            errorStr += "Number training days must be greater than 10"
            context_dict['post_error'] = True
        else:
            modelParams['trainingDays'] = numTrainingDays

        try:
            featureLength = int(request.POST.get("featureLength"))
        except:
            featureLength = -1
        if featureLength < 1:
            errorStr += "Feature Length must be greater than 1"
            context_dict['post_error'] = True
        else:
            modelParams['featureLength'] = featureLength

        try:
            validationPercent = int(request.POST.get("validationPercent"))
        except:
            validationPercent = -1
        if validationPercent < 1:
            errorStr += "Validation Percent must be greater than 0"
            context_dict['post_error'] = True
        else:
            modelParams['validationPercent'] = validationPercent

        try:
            positiveLabelValue = float(request.POST.get("positiveLabelValue"))
        except:
            positiveLabelValue = -1
        if positiveLabelValue < 0:
            errorStr += "Positive Label Value must be greater than 0"
            context_dict['post_error'] = True
        else:
            modelParams['positiveLabelValue'] = positiveLabelValue

        try:
            holdDays = int(request.POST.get("holdDays"))
        except:
            holdDays = -1
        if holdDays < 0:
            errorStr += "Number of hold days must be greater than 0"
            context_dict['post_error'] = True
        else:
            modelParams['holdDays'] = holdDays



        RunTensorDNNModelBuilder.delay(modelParams)
        startingModelRun = True


    if(CheckStateFileRunning(60) or startingModelRun):
        context_dict['LoadingDiv'] = "<div class=\"loader\"></div>"
    else:
        context_dict['LoadingDiv'] = ""

    context_dict['error_message'] = errorStr
    context_dict.update(modelParams)

    stockIndexItems = ''
    for eachInd in stockIndices:
        if(eachInd == indexName):
            stockIndexItems += "<option selected value="+eachInd+">" + eachInd + "</option>\n"
        else:
            stockIndexItems += "<option value="+ eachInd +">" + eachInd + "</option>\n"
    context_dict['DataIndices'] = stockIndexItems



    return render(request, 'userPortfolio/build_model.html', context_dict)


@shared_task
def RunTensorDNNModelBuilder(modelParams):
    print(settings.MODELS_STATE_FILE)
    file = open(settings.MODELS_STATE_FILE, 'w')
    timestr = datetime.datetime.now().strftime(settings.DATETIMESTRFORMAT)
    file.write(timestr)
    tdnnBuilder = userPortfolio.TensorDNNModel.TensorDNNModelBuilder(modelParams)
    tdnnBuilder.BeginNewBuild()


def CheckStateFileRunning(maxTimeDelta):
    if os.path.exists(settings.MODELS_STATE_FILE):
        file = open(settings.MODELS_STATE_FILE, 'r')
        fileData = file.read()
        try:
            fileTime = datetime.datetime.strptime(fileData, settings.DATETIMESTRFORMAT)
            currentTime = datetime.datetime.now()
        except:
            return False
        timeDiff = currentTime - fileTime
        timeDiffSecs = timeDiff.total_seconds()
        print("Time Difference *** ")
        print(timeDiffSecs)

        if(timeDiffSecs > maxTimeDelta):
            return False
        return True

    else:
        return False


