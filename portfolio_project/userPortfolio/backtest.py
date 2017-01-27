import datetime
import sys,os
from userPortfolio.models import UserTransactionsModel, UserPortfolioModel, AllStocksModel
from django.db import IntegrityError
from django.conf import settings
import csv, hashlib
from pandas_datareader import data
import pandas, urllib2, csv
from pytz import timezone
from dateutil.relativedelta import relativedelta
from math import pi
from talib.abstract import *
from userPortfolio import strategies
import userPortfolio.backend as backend
import matplotlib.pyplot as plt


class CrossValidation:

    def __init__(self, strategy, type='snp500', num_days=365, money_per_stock=500, variable='', var_range=[0,1,1]):
        self.strategy = strategy
        self.type = type
        self.num_days = num_days
        self.money = money_per_stock
        self.variable = variable
        self.variable_range = var_range
        self.time_limit = 15

        results_list = []
        var_vals = []
        profit_ratios  = []
        for i in range(var_range[0], var_range[1], var_range[2]):
            eachResult = self.run_with_var(i)
            flat_result = [item for sublist in eachResult for item in sublist]

            losses = 0
            nobuys = 0
            profits = 0
            for eachEl in flat_result:
                if eachEl < 0:
                    losses = losses + 1
                if eachEl == 0:
                    nobuys = nobuys + 1
                if eachEl > 0:
                    profits = profits + 1
            result_dic = {'var':self.variable, 'var_val':i, 'profits':profits, 'losses':losses, 'nobuys':nobuys, 'profit_ratio' : float(profits)/float(profits+losses)}
            results_list.append(result_dic)
            var_vals.append(result_dic['var_val'])
            profit_ratios.append(result_dic['profit_ratio'])
            print(result_dic)

        plt.plot(var_vals, profit_ratios)
        plt.show()
        self.results = (var_vals, profit_ratios)




    def run_with_var(self, value):
        #Get dates using AAPL stock as reference
        index_stock = backend.get_historical_stock_data('AAPL')
        end_loc = index_stock.index.get_loc(index_stock.index[-1]) - self.time_limit - 30
        start_loc = end_loc-(self.num_days*self.time_limit)


        all_results = []
        currentTimeUtc = datetime.datetime.now(timezone('UTC'))
        nycTime = currentTimeUtc.astimezone(timezone('US/Eastern'))
        end_date_t = (nycTime-datetime.timedelta(days=self.time_limit+1))
        end_date = end_date_t.date()
        print("end date is "+str(end_date))
        start_date = (end_date_t-datetime.timedelta(days=self.num_days)).date()
        print("start date is "+str(start_date))
        current_date = start_date
        current_loc = start_loc
        while current_loc < end_loc:
            # date = current_date
            # current_date_t = datetime.datetime.combine(current_date, datetime.time.min)
            # current_date = (current_date_t+datetime.timedelta(days=1)).date()
            # if date.isoweekday() > 5:
            #     print("Weekend: Skipping")
            #     continue
            date = index_stock.index[current_loc]
            current_loc = current_loc + self.time_limit
            strat = self.strategy(date=date, type=self.type)
            setattr(strat, self.variable, value)
            selectedStocks = strat.run()
            current_results = self.process_picks(selectedStocks)
            all_results.append(current_results)
        return all_results

    def process_picks(self, selectedStocks):
        cost = 10
        tax = 30
        results = []
        #print(selectedStocks)
        for index, eachStock in selectedStocks.iterrows():
            stock_data = backend.get_historical_stock_data(eachStock.stockName)
            date = eachStock.date
            loc = stock_data.index.get_loc(date)+1
            buyPrice = (stock_data.iloc[loc].high + stock_data.iloc[loc].low)/2
            num_stocks = self.money/buyPrice

            #unable to buy
            if num_stocks < 1:
                results.append(0)
                continue

            #profit
            for i in range(self.time_limit):
                loc = loc + 1
                current_data = stock_data.iloc[loc]
                currentPrice = (current_data.high + current_data.low)/2
                if currentPrice > buyPrice:
                    diff = (currentPrice-buyPrice)*num_stocks
                    profit = diff - (0.3*diff)
                    if profit >= cost:
                        results.append(1)
                        break

            #loss
            results.append(-1)
        print("Day Results "+str(results))
        return results







#
