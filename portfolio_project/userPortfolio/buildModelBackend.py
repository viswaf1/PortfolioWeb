import datetime
import sys,os
from django.shortcuts import render, render_to_response
from django.http import HttpResponse
from userPortfolio.models import UserTransactionsModel, UserPortfolioModel, \
        AllStocksModel, SNP500Model, UserProfile, ModelFileData
from django.db import IntegrityError
from django.conf import settings
from django.utils import timezone as djangotimezone
import csv, hashlib, pickle
from pandas_datareader import data
import pandas, urllib.request, urllib.error, urllib.parse, csv
from pytz import timezone
from dateutil.relativedelta import relativedelta
import datetime, random, string
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.resources import CDN
from bokeh.models import Legend, LegendItem
from bokeh.embed import components
from bokeh.palettes import Category20c
from bokeh.transform import cumsum
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

        print(settings.MODELS_STATE_FILE)
        file = open(settings.MODELS_STATE_FILE, 'w')
        timestr = datetime.datetime.now().strftime(settings.DATETIMESTRFORMAT)
        file.write(timestr)

        currTime = datetime.datetime.now()
        nameStr = currTime.strftime("%Y-%m-%d-%H-%M-%S")
        randomStr = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(5)])
        modelpath = nameStr + randomStr
        modelpath = os.path.join(settings.MODELS_DIR, modelpath)

        modelEntry = ModelFileData.objects.create(type="TENSOR_DNN",
                                                  path=modelpath, datetime=currTime,
                                                  state="INIT")
        modelEntry.save()

        modelParams['modelId'] = modelEntry.id

        RunTensorDNNModelBuilder.delay(modelParams)
        startingModelRun = True

    # modelCurrentlyRunning = CheckStateFileRunning(60)
    # if modelCurrentlyRunning:
    try:
        currModel = ModelFileData.objects.all().order_by("-id")[0]
        if(currModel.state != 'BUILDDONE'):
            context_dict['modelId'] = currModel.id
            modelCurrentlyRunning = True
    except:
        print("ERRRO NO MODEL ID")
        modelCurrentlyRunning = False



    if(modelCurrentlyRunning or startingModelRun):
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

    tdnnBuilder = userPortfolio.TensorDNNModel.TensorDNNModelBuilder(modelParams)
    tdnnBuilder.BeginNewBuild()


# def CheckStateFileRunning(maxTimeDelta):
    # if os.path.exists(settings.MODELS_STATE_FILE):
    #     file = open(settings.MODELS_STATE_FILE, 'r')
    #     fileData = file.read()
    #     try:
    #         fileTime = datetime.datetime.strptime(fileData, settings.DATETIMESTRFORMAT)
    #         currentTime = datetime.datetime.now()
    #     except:
    #         return False
    #     timeDiff = currentTime - fileTime
    #     timeDiffSecs = timeDiff.total_seconds()
    #     print("Time Difference *** ")
    #     print(timeDiffSecs)
    #
    #     if(timeDiffSecs > maxTimeDelta):
    #         return False
    #     return True
    #
    # else:
    #     return False

def GetModelState(request):
    if request.method == 'POST' :
        modelId = request.POST.get("ModelId")
        try:
            modelId = int(modelId)
        except:
            return HttpResponse("")

    elif request.method == 'GET' :
        modelId = request.GET.get("ModelId")
        try:
            modelId = int(modelId)
        except:
            return HttpResponse("")

    try:
        trainModelObj = ModelFileData.objects.get(pk=modelId)
    except:
        return HttpResponse("")
    return HttpResponse(trainModelObj.state)


def RenderModelProgress(request):
    stateOrder = ['INIT', 'FEATUREGEN', 'NETWORKTRAIN']
    errorStr = ''
    context_dict = {'post_error' : False, 'builderRunning' : False}
    if request.method == 'POST' :
        modelId = request.POST.get("ModelId")
        try:
            modelId = int(modelId)
        except:
            errorStr += "Error Invalid Model Id."
            context_dict['post_error'] = True
            return render(request, 'userPortfolio/model_progress.html', context_dict)
    elif request.method == 'GET' :
        modelId = request.GET.get("ModelId")
        try:
            modelId = int(modelId)
        except:
            errorStr += "Error Invalid Model Id."
            context_dict['post_error'] = True
            return render(request, 'userPortfolio/model_progress.html', context_dict)
    else:
        request.modelId = -1
        return render(request, 'userPortfolio/model_progress.html', context_dict)

    try:
        trainModelObj = ModelFileData.objects.get(pk=modelId)
    except:
        # errorStr += "Error Invalid Model Id."
        # context_dict['post_error'] = True
        context_dict['builderRunning'] = False
        return render(request, 'userPortfolio/model_progress.html', context_dict)

    context_dict['modelId'] = modelId
    context_dict['builderRunning'] = True

    stateIndex = stateOrder.index(trainModelObj.state)
    if trainModelObj.state == 'INIT':
        context_dict['state'] = "Model Builder Initializing"

    elif trainModelObj.state == 'FEATUREGEN':
        context_dict['state'] = "Generating Features and Labels"

    elif trainModelObj.state == 'NETWORKTRAIN':
        context_dict['state'] = "Training Neural Network"

    if stateIndex > stateOrder.index('FEATUREGEN'):
        statsPath = os.path.join(trainModelObj.path, settings.FEATURE_STAT_FILENAME)
        validStatsPath = os.path.join(trainModelObj.path, settings.FEATURE_VALIDATION_STAT_FILENAME)
        fileReadSuccess = False
        try:
            print("Reading feature stats file from "+statsPath)
            statsFile = open( statsPath, "rb" )
            print("Reading validation feature stats file from " + validStatsPath)
            validStatsFile = open(validStatsPath, "rb")
            fileReadSuccess = True
        except:
            pass
        if fileReadSuccess:
            featureStats = pickle.load(statsFile)
            validFeatureStats = pickle.load(validStatsFile)
            statsStr = ''
            statsStr += "<p>Model Directory : "+str(trainModelObj.path)+"/TensorModel</p>\n"
            statsStr += "<p>Number of Training Features : " + str(featureStats['NumFeatures']) + "</p>\n"
            statsStr += "<p>Number of valiation Features : " + str(validFeatureStats['NumFeatures']) + "</p>\n"
            trainPlot = BokehDrawLabelPieChart(featureStats, "Training")
            validPlot = BokehDrawLabelPieChart(validFeatureStats, "Validation")
            plot = row(trainPlot, validPlot)
            script, div = components(plot, CDN)
            # import pdb;pdb.set_trace()
            context_dict['featureStatDiv'] = div
            context_dict['featureStatScript'] = script
            context_dict['featureStatStr'] = statsStr
            context_dict['hasFeatureStats'] = True


            # import pdb;pdb.set_trace()

    return render(request, 'userPortfolio/model_progress.html', context_dict)


def BokehDrawLabelPieChart(featureStats, title):
    labelStats = {}
    for eachKey in featureStats['LabelCounts'].keys():
        labelStats[str(eachKey)] = featureStats['LabelCounts'][eachKey]


    data = pandas.Series(labelStats).reset_index(name='value').rename(columns={'index': 'label'})
    data['angle'] = data['value'] / data['value'].sum() * 2 * pi

    if len(labelStats) == 1:
        data['color'] = ['#F44242']
    elif len(labelStats) == 2:
        data['color'] = ['#F44242', '#1357C4']
    else:
        data['color'] = Category20c[len(labelStats)]

    p = figure(plot_height=200, plot_width=350, title=title, toolbar_location=None,
               tools="hover", tooltips="@label: @value", x_range=(-0.5, 1.0))

    p.wedge(x=0, y=1, radius=0.3,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend='label', source=data)

    p.axis.axis_label = None
    p.axis.visible = False
    p.grid.grid_line_color = None
    # script, div = components(p, CDN)

    return p
