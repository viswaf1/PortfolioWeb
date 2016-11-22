import datetime
import sys,os
from userPortfolio.models import UserTransactionsModel, UserPortfolioModel, AllStocksModel, SNP500Model
from django.db import IntegrityError
from django.conf import settings
import csv, hashlib
from pandas_datareader import data
import pandas, urllib2, csv
from pytz import timezone
from dateutil.relativedelta import relativedelta
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.resources import CDN
from bokeh.models import Legend, LinearAxis
from bokeh.embed import components
from math import pi
from talib.abstract import *


def buy_stock(user, stockName, buyPrice, numberOfStocks, buyDate):
    #check if stockName is present in portfolio_table
    portQs = UserPortfolioModel.objects.filter(username = user, stockName = stockName)
    amount = buyPrice*numberOfStocks
    if len(portQs) > 0:
        portObj = portQs[0]
        portObj.moneyInvested = portObj.moneyInvested + amount
        portObj.numberOfStocks = portObj.numberOfStocks + numberOfStocks
        portObj.save()
        portId = portObj.portfolioId
    else:
        #generate uique portfolio id for this transactions
        portId = hashlib.md5(stockName+str(buyPrice)+buyDate.strftime("%B%d,%Y")).hexdigest()
        UserPortfolioModel.objects.create(username = user, portfolioId = portId, \
        stockName = stockName, moneyInvested = amount, numberOfStocks = numberOfStocks)

    #Add transaction with portfolioId to transaction tables
    UserTransactionsModel.objects.create(username = user, portfolioId = portId, \
    stockName = stockName, buyDate = buyDate, buyPrice = buyPrice, \
    numberOfStocksBought = numberOfStocks)

    return 1

def sell_stock(user, stockName, stockPrice, numOfStocks, transactionDate):
    portQs = UserPortfolioModel.objects.filter(username = user, stockName = stockName)
    amount = stockPrice*numOfStocks

    portObj = portQs[0]
    newMoneyInvested = portObj.moneyInvested - amount
    newNumberOfStocks = portObj.numberOfStocks - numOfStocks
    if newNumberOfStocks < 0:
        return 0
    portObj.moneyInvested = newMoneyInvested
    portObj.numberOfStocks = newNumberOfStocks
    portObj.save()
    portId = portObj.portfolioId

    UserTransactionsModel.objects.create(username = user, portfolioId = portId, \
    stockName = stockName, sellDate = transactionDate, sellPrice = stockPrice, \
    numberOfStocksSold = numOfStocks)

    if portObj.numberOfStocks == 0:
        portObj.delete()
    return 1

def get_quote_today(symbol):
    YAHOO_TODAY="http://download.finance.yahoo.com/d/quotes.csv?s=%s&f=sd1ohgl1vl1"
    response = urllib2.urlopen(YAHOO_TODAY % symbol)
    reader = csv.reader(response, delimiter=",", quotechar='"')
    for row in reader:
        if row[0] == symbol:
            return row

def append_today_quote(dataFrame, endDate, stockName):
    frameEndDate = dataFrame.index.max().to_pydatetime().date()
    if(frameEndDate < endDate):
        df = pandas.DataFrame(index=pandas.DatetimeIndex(start=endDate, end=endDate, freq="D"),
                  columns=["open", "high", "low", "close", "volume", "adj close"],
                  dtype=float)
        row = get_quote_today(stockName)
        df.ix[0] = map(float, row[2:])
        dataFrame = dataFrame.append(df)

def get_historical_stock_data(stockName):
    stockFile = stockName+'_historical.csv'
    stockFileDir = settings.BASE_DIR+'/stock_data/'
    stockFilePath = stockFileDir+stockFile


    currentTimeUtc = datetime.datetime.now(timezone('UTC'))
    nycTime = currentTimeUtc.astimezone(timezone('US/Eastern'))
    nycCloseTime = nycTime.replace(hour=16, minute=0, second=0, microsecond=0)
    nycCloseTimeUtc = nycCloseTime.astimezone(timezone('UTC'))

    queryStartDate = '1/1/1990'

    if(currentTimeUtc > nycCloseTimeUtc):
        queryEndDate = nycTime.date()
    else:
        nycTimeYesterday = nycTime
        nycTimeYesterday = nycTimeYesterday - datetime.timedelta(days=1)
        queryEndDate = nycTimeYesterday.date()

    if queryEndDate.isoweekday() > 5:
        diff = queryEndDate.isoweekday() - 5
        temp_date_time = datetime.datetime.combine(queryEndDate, datetime.time.min)
        queryEndDate = (temp_date_time - datetime.timedelta(days=diff)).date()

    queryEndDateStr = queryEndDate.strftime("%m/%d/%Y")

    #queryEndDate = datetime.date.today().strftime("%m/%d/%Y")

    if not os.path.exists(stockFileDir):
        os.makedirs(stockFileDir)
    #Check if stock file exists
    if not os.path.exists(stockFilePath):
        #try:
        stockFrame = data.DataReader(stockName, 'yahoo', queryStartDate, queryEndDateStr)
        stockFrame.columns = map(str.lower, stockFrame.columns)

        append_today_quote(stockFrame, queryEndDate, stockName)

        stockFrame.to_csv(stockFilePath, sep=',')
        # except:
        #     stockFrame = pandas.DataFrame()

    else:
        stockFrame = pandas.read_csv(stockFilePath, delimiter = ',', index_col=0, parse_dates=True)
        frameEndDate = stockFrame.index.max().to_pydatetime().date()
        currentDate = datetime.date.today()
        if(frameEndDate < queryEndDate):
            print frameEndDate
            print currentDate
            print "Not up to date.... updating"
            try:
                tempStockFrame = data.DataReader(stockName, 'yahoo', frameEndDate.strftime("%m/%d/%Y"))
                tempStockFrame.columns = map(str.lower, tempStockFrame.columns)
                if not frameEndDate == tempStockFrame.index.max().to_pydatetime().date():
                    stockFrame = pandas.concat([stockFrame, tempStockFrame])
                    append_today_quote(stockFrame, queryEndDate, stockName)
                    stockFrame.to_csv(stockFilePath, sep=',')
            except:
                stockFrame = pandas.DataFrame()

    return stockFrame


def render_stock_data(stockName):
    fancyRed = '#F44242'
    fancyBlue = '#1357C4'
    fancyGreen = '#7AF442'
    stockFrame = get_historical_stock_data(stockName)
    stock_data = stockFrame.tail(150)
    mids = (stock_data.open + stock_data.close)/2
    spans = abs(stock_data.close-stock_data.open)

    inc = stock_data.close > stock_data.open
    dec = stock_data.open > stock_data.close
    w = 12*60*60*1000 # half day in ms

    TOOLS = "pan,wheel_zoom"
    plotWidth = 800
    smallplotHeight = 100
    legendLoc = (60, smallplotHeight-40)
    xRange = (stock_data.tail(60).index.min(), stock_data.tail(60).index.max())
    yRange = (stock_data.tail(60).min(), stock_data.tail(60).max())

    candles = figure(x_axis_type="datetime", tools=TOOLS, plot_width=plotWidth,
    plot_height=400, x_range=xRange, title = "Candlestick")
    candles.xaxis.major_label_orientation = pi/4
    #chadles.grid.grid_line_alpha=0.3

    candles.segment(stock_data.index, stock_data.high, stock_data.index, stock_data.low, color="black")
    candles.rect(stock_data.index[inc], mids[inc], w, spans[inc], fill_color=fancyGreen, line_color="black")
    candles.rect(stock_data.index[dec], mids[dec], w, spans[dec], fill_color=fancyRed, line_color="black")


    volume_plot = figure(x_axis_type="datetime", tools=TOOLS, plot_width=plotWidth,
    plot_height=smallplotHeight,  x_range=candles.x_range)
    volume_plot.vbar(x=stock_data.index, top=stock_data.volume, width=w, bottom=0,
    color=fancyBlue)
    volume_plot.xaxis.visible = False

    sar_data = SAR(stock_data)
    print sar_data
    mcl1 = candles.line(sar_data.index, sar_data, line_width=1,
    line_color=fancyRed)

    #plot the MACD indicator
    macd_data = MACD(stock_data, fastperiod=12, slowperiod=26, signalperiod=9)
    macd_range = calc_range((macd_data.macdsignal, macd_data.macd), 60)
    macd_plot = figure(x_axis_type="datetime", tools=TOOLS, plot_width=plotWidth,
    plot_height=smallplotHeight,  x_range=candles.x_range, y_range=macd_range)
    macd_plot.xaxis.visible = False

    #macd_plot.Bar()
    mcl1 = macd_plot.line(macd_data.index, macd_data.macdsignal, line_width=2,
    line_color=fancyRed)
    mcl2 = macd_plot.line(macd_data.index, macd_data.macd, line_width=2, line_color=fancyBlue)

    legend = Legend(legends=[
    ("MACD",   [mcl2]),
    ("MACD signal", [mcl1]),
], location=legendLoc)
    macd_plot.add_layout(legend)

    #ADX and +DM and -DM
    adx_data = ADX(stock_data, timeperiod=14)
    plusdm_data = PLUS_DM(stock_data, timeperiod=14)
    minusdm_data = MINUS_DM(stock_data, timeperiod=14)
    adx_range = calc_range((adx_data, plusdm_data, minusdm_data), 60)
    print "ADX Range is =============="
    print adx_range

    adx_plot = figure(x_axis_type='datetime', tools=TOOLS, plot_width=plotWidth,
    plot_height=smallplotHeight, x_range=candles.x_range, y_range=adx_range)
    adx_plot.xaxis.visible = False


    adl1 = adx_plot.line(adx_data.index, adx_data, line_width=2, line_color="black")
    adl2 = adx_plot.line(plusdm_data.index, plusdm_data, line_width=2, line_color=fancyBlue)
    adl3 = adx_plot.line(plusdm_data.index, minusdm_data, line_width=2, line_color=fancyRed)
    legend = Legend(legends=[
    ("ADX",   [adl1]),
    ("+DM", [adl2]),
    ("-DM", [adl3]),
], location=legendLoc)
    adx_plot.add_layout(legend)

    #slow stochastic
    stoc_data = STOCH(stock_data, fastk_period=5, slowk_period=3,
    slowk_matype=0, slowd_period=3, slowd_matype=0)

    stoc_plot = figure(x_axis_type='datetime', tools=TOOLS, plot_width=plotWidth,
    plot_height=smallplotHeight, x_range=candles.x_range)
    stoc_plot.xaxis.visible = False

    stl1 = stoc_plot.line(stoc_data.index, stoc_data.slowk, line_width=2, line_color='black')
    stl2 = stoc_plot.line(stoc_data.index, stoc_data.slowd, line_width=2, line_color=fancyRed)
    legend = Legend(legends=[
    ("%K",   [stl1]),
    ("%D", [stl2]),
], location=legendLoc)
    stoc_plot.add_layout(legend)

    #CCI plot
    cci_data = CCI(stock_data, timeperiod=14)
    cci_plot = figure(x_axis_type='datetime', tools=TOOLS, plot_width=plotWidth,
    plot_height=smallplotHeight, x_range=candles.x_range)

    ccl1 = cci_plot.line(cci_data.index, cci_data, line_width=2, line_color='black')
    legend = Legend(legends=[
    ("CCI",   [ccl1]),
], location=legendLoc)
    cci_plot.add_layout(legend)

    #RSI plot
    rsi_data = RSI(stock_data, timeperiod=14)
    rsi_plot = figure(x_axis_type='datetime', tools=TOOLS, plot_width=plotWidth,
    plot_height=smallplotHeight, x_range=candles.x_range)

    rsil1 = rsi_plot.line(rsi_data.index, rsi_data, line_width=2, line_color='black')
    legend = Legend(legends=[
    ("RSI",   [rsil1]),
], location=legendLoc)
    rsi_plot.add_layout(legend)

    plot = column(candles, volume_plot, macd_plot, adx_plot, stoc_plot, cci_plot, rsi_plot)
    #Store components
    script, div = components(plot, CDN)

    return (script, div)

def calc_range(data, num_vals):
    maxs = []
    mins = []
    for each in data:
        maxs.append(each.tail(num_vals).max())
        mins.append(each.tail(num_vals).min())
    return((min(mins), max(maxs)))

def import_stocklist_csv():
    nasdaq_csv_path = '/home/teja/NASDAQ.csv'
    nyse_csv_path = '/home/teja/NYSE.csv'

    with open(nasdaq_csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print row['Symbol']
            try:
                AllStocks.objects.get_or_create(stockName = row['Symbol'], \
                name = row['Name'], sector = row['Sector'], industry = row['Industry'], \
                market = 'NASDAQ')
            except IntegrityError as e:
                print "stock "+row['Symbol']+" Already Present"

    with open(nyse_csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print row['Symbol']
            try:
                AllStocks.objects.get_or_create(stockName = row['Symbol'], \
                name = row['Name'], sector = row['Sector'], industry = row['Industry'], \
                market = 'NYSE')
            except IntegrityError as e:
                print "stock "+row['Symbol']+" Already Present"

def import_stocklist_snp():
    snp_csv_path = '/home/teja/snp500.csv'
    with open(snp_csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print row['Symbol']
            try:
                SNP500Model.objects.get_or_create(stockName = row['Symbol'], \
                name = row['Name'], sector = row['Sector'], industry = 'None', \
                market = 'NASDAQ')
            except IntegrityError as e:
                print "stock "+row['Symbol']+" Already Present"
