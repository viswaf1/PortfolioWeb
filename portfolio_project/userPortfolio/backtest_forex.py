import datetime, time
import sys,os, random, string, math, cPickle, subprocess, multiprocessing, logging
from django.contrib.auth.models import User
from userPortfolio.models import UserTransactionsModel, UserPortfolioModel
from userPortfolio.models import AllStocksModel, BacktestDataModel, UserProfile
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
import talib
import numpy as np
from operator import itemgetter


class BackTest:

    def __init__(self):
        self.starting_amount = 10000.00
        self.stoploss = 1
        self.target = 5
        self.password = 'backtest'
        self.stock_max = 10000.00
        self.stock_min = 8000.00
        self.max_num_days = 10
        self.stop_loss_percent = 1.0
        self.profit_target_percent = 5.0
        self.percent_flag = True
        self.percent_flag_target = True
        self.stop_loss_immediate = False
        self.stop_atr_percent = 2.0
        self.target_atr_percent = 2.0
        self.atr_time_period = 10
        self.brokerage = 10.0
        self.tax_percent = 30.0
        self.max_num_stocks = 2
        self.sar_acceleration = 0.02
        self.sar_maximum = 0.2
        # self.picks_file = 'picked_stocks_voting3_filtered.txt'
        self.picks_file = 'picked_stocks_forex_future_test.txt'

    def load_picks_dictionary(self):
        try:
            self.picks_dictionary = cPickle.load(open(self.picks_file, 'rb'))
        except Exception as ex:
            print(str(ex))
            self.picks_dictionary = {}

    def picks_callback(self, results):
        picks, pick_date = results
        date_str = pick_date.strftime('%m%d%Y')
        self.load_picks_dictionary()
        self.picks_dictionary[date_str] = picks
        cPickle.dump(self.picks_dictionary, open(self.picks_file, 'wb'))

    def run_picks(self, num_days, test_start_date=None, reset=False):
        if reset:
            picks_dic = {}
        else:
            try:
                self.picks_dictionary = cPickle.load(open(self.picks_file, 'rb'))
            except Exception as ex:
                print(str(ex))
                self.picks_dictionary = {}

        stockData = backend.StockData.Instance()
        index_data = stockData.get_historical_stock_data('^GSPC')

        start_ind = None
        end_ind = None
        if test_start_date is None:
            end_ind = len(index_data.index)-1
            start_ind = end_ind-num_days
        else:
            start_ind = self.find_index_greater(test_start_date)
            end_ind = start_ind+num_days
        mpl = multiprocessing.log_to_stderr()
        mpl.setLevel(logging.INFO)


        next_day_transactions = []
        # pool = multiprocessing.Pool(3)
        for dayInd in range(start_ind, end_ind+1):
            current_date = index_data.index[dayInd].date()
            current_date_str = current_date.strftime('%m%d%Y')
            if current_date_str in self.picks_dictionary:
                continue
            print("calling")
            #pool.apply_async(pick_subprocess, args=(current_date,), callback=self.picks_callback)
            self.picks_callback(pick_subprocess((current_date,)))
        # time.sleep(200)
        # pool.close()
        # pool.join()




    def get_picks_date(self, pick_date):
        date_str = pick_date.strftime('%m%d%Y')
        if date_str in self.picks_dictionary:
            picked_stocks =  self.picks_dictionary[date_str]
        else:
            self.picks_callback(pick_subprocess((pick_date,)))
            picked_stocks = self.picks_dictionary[date_str]
        #return picked_stocks
        return self.post_process_picks(picked_stocks, pick_date)

    def post_process_picks(self, picks, pick_date):
        post_picks = []
        pick_scores = []
        for stock in picks:
            stock_data = backend.StockData.Instance().get_historical_stock_data(stock)
            stock_data = stock_data.ix[:pick_date]
            high = stock_data['high'].values
            low = stock_data['low'].values
            close = stock_data['close'].values
            mom3 = talib.MOM(close[-100:], timeperiod=3)
            ma5 = talib.MA(close, timeperiod=2)
            macd, macdsignal, macdhist = talib.MACD(close)
            # disparity = close[-1]/ma5[-1]
            atr = talib.ATR(high[-100:], low[-100:], close[-100:], timeperiod=20)
            if macdhist[-1] > 0:
            #if close[-1] > 100:
                #pick_scores.append({'stock':stock, 'score':atr[-1]})
                pick_scores.append({'stock':stock, 'score':mom3[-1]})
                post_picks.append(stock)
        #return post_picks
        sorted_stocks = sorted(pick_scores, key=itemgetter('score'), reverse=True)
        return [x['stock'] for x in sorted_stocks]


    def run_backtest(self, num_days, test_start_date=None, reset_picks = False):
        if reset_picks:
            self.picks_dictionary = {}
        else:
            self.load_picks_dictionary()

        user_name = 'testuser'
        if len(User.objects.filter(username=user_name)):
            User.objects.get(username=user_name).delete()
        backtest_user = User.objects.create_user(user_name, 'tester@test.com', self.password)
        backtest_user.save()

        backtest_user_profile = UserProfile(user=backtest_user,
        moneyAvailable=self.starting_amount)
        backtest_user_profile.save()

        #Get the snp500 index stock and pick the starting and ending dates using the user input
        stockData = backend.StockData.Instance()
        index_data = stockData.get_historical_stock_data('^GSPC')

        start_ind = None
        end_ind = None
        if test_start_date is None:
            end_ind = len(index_data.index)-1
            start_ind = end_ind-num_days

        else:
            start_ind = self.find_index_greater(test_start_date)
            end_ind = start_ind+num_days

        all_liss = []
        next_day_transactions = []
        for dayInd in range(start_ind, end_ind+1):
            current_date = index_data.index[dayInd].date()

            for eachNxtTrans in next_day_transactions:
                if eachNxtTrans['action'] == 'buy':
                    stock_name = eachNxtTrans['stock']
                    stock_data = backend.StockData.Instance().get_historical_stock_data(stock_name)
                    try:
                        num_stocks = int(math.floor(eachNxtTrans['money_available']/stock_data.ix[current_date].open))
                        money_invested = num_stocks*stock_data.ix[current_date].open
                        stop_loss = self.calculate_stop_loss(stock_data.ix[:current_date], initial=True, timeperiod=self.atr_time_period, atr_percent=self.stop_atr_percent)
                    #stop_target = ((money_invested)+(money_invested*(self.profit_target_percent/100)))/num_stocks
                        stop_target = self.calculate_stop_target(stock_data.ix[:current_date], initial=True, timeperiod=self.atr_time_period, atr_percent=self.target_atr_percent)

                        backend.buy_stock(backtest_user, eachNxtTrans['stock'], stock_data.ix[current_date].open,
                            num_stocks, current_date, stop_loss, stop_target)
                    except Exception as e:
                        print e
                        continue
                if eachNxtTrans['action'] == 'sell':
                    stock_data = backend.StockData.Instance().get_historical_stock_data(eachNxtTrans['stock'])
                    try:
                        backend.sell_stock(backtest_user, eachNxtTrans['stock'], stock_data.ix[current_date].open,
                        eachNxtTrans['num_stocks'], current_date, self.brokerage, self.tax_percent, reason=eachNxtTrans['reason'])
                    except Exception as e:
                        print e
                        continue
            today_buys = [x['stock'] for x in next_day_transactions if x['action'] == 'buy']
            next_day_transactions = []

            all_picked_stocks = self.get_picks_date(current_date)
            print all_picked_stocks

            portQs = UserPortfolioModel.objects.filter(username=backtest_user)
            for eachPort in portQs:
                stk = eachPort.stockName
                stock_data = backend.StockData.Instance().get_historical_stock_data(stk)
                try:
                    current_open = stock_data.ix[current_date].open
                    current_min = stock_data.ix[current_date].low
                    current_max = stock_data.ix[current_date].high
                    current_close = stock_data.ix[current_date].close
                except Exception as e:
                    print e
                    continue
                port_trans = UserTransactionsModel.objects.filter(portfolioId=eachPort.portfolioId)
                last_trans_date = port_trans[0].buyDate
                last_trans_index = index_data.index.get_loc(last_trans_date.date())

                if index_data.index.get_loc(current_date) - last_trans_index >= self.max_num_days:
                    backend.sell_stock(backtest_user, stk, current_open, -1, current_date,
                    self.brokerage, self.tax_percent, reason='time limit')
                    continue

                elif stk not in today_buys and current_open < eachPort.stopLoss:
                    backend.sell_stock(backtest_user, stk, current_open, -1, current_date,
                    self.brokerage, self.tax_percent, reason='open loss')
                #
                # elif current_min < eachPort.minStopLoss or current_open < eachPort.minStopLoss:
                #     backend.sell_stock(backtest_user, stk, eachPort.minStopLoss, -1, current_date,
                #      self.brokerage, self.tax_percent, reason='min stop loss')
                #
                elif current_min < eachPort.stopLoss:
                    # backend.sell_stock(backtest_user, stk, eachPort.stopLoss, -1, current_date,
                    #   self.brokerage, self.tax_percent, reason='stop loss')
                    if not self.stop_loss_immediate:
                        next_day_transactions.append({'stock':stk, 'num_stocks':-1, 'reason':'stop loss', 'action':'sell'})
                    else:
                        backend.sell_stock(backtest_user, stk, eachPort.stopLoss, -1, current_date,
                        self.brokerage, self.tax_percent, reason='stop_loss_immediate')
                        continue
                # #
                elif current_max >= eachPort.stopTarget:
                    backend.sell_stock(backtest_user, stk, eachPort.stopTarget, -1, current_date,
                    self.brokerage, self.tax_percent, reason='target')
                #
                else:
                    new_stop_loss = self.calculate_stop_loss(stock_data.ix[:current_date], timeperiod=self.atr_time_period, atr_percent=self.stop_atr_percent)
                    if new_stop_loss > eachPort.stopLoss:
                        eachPort.stopLoss = new_stop_loss
                        eachPort.save()
                #=======================================
                #     else:
                #         eachPort.stopTarget = self.calculate_stop_target(stock_data.ix[:current_date])
                #         eachPort.save()

                # if stk not in all_picked_stocks:
                #     next_day_transactions.append({'stock':stk, 'num_stocks':-1, 'reason':'Not in future picks', 'action':'sell'})
                #     continue


                #check if the trend is a downward trend and sell it on the next open
                stock_data_slice = stock_data.ix[:current_date]
                prev_closes = stock_data_slice['close'].values
                macd, macdsignal, macdhist = talib.MACD(prev_closes)
                sar_data = talib.SAR(stock_data_slice['high'].values, stock_data_slice['low'].values, acceleration=self.sar_acceleration, maximum=self.sar_maximum)
                prev_closes = prev_closes[-5:]
                # close_diffs = []
                # for i in range(len(prev_closes)-1, 0, -1):
                #     close_diffs.append(prev_closes[i]-prev_closes[i-1])
                # if all(i <= 0 for i in close_diffs):
                #     next_day_transactions.append({'stock':stk, 'num_stocks':-1, 'reason':'downward trend', 'action':'sell'})
                #     continue
                chl = calc_chandelier_long(stock_data_slice)
                if prev_closes[-1] < chl:
                    next_day_transactions.append({'stock':stk, 'num_stocks':-1, 'reason':'macd crossover', 'action':'sell'})
                    continue

                # close_change = (current_close*eachPort.numberOfStocks - eachPort.moneyInvested)
                # if close_change < 0 and (close_change*-1/eachPort.moneyInvested)*100 > 2.0:
                #     next_day_transactions.append({'stock':stk, 'num_stocks':-1, 'reason':'close change 0.5%', 'action':'sell'})
                #     continue

            portfolio_stocks = []
            portQs = UserPortfolioModel.objects.filter(username=backtest_user)
            for eachPort in portQs:
                portfolio_stocks.append(eachPort.stockName)

            picked_stocks = []
            for each_picked in all_picked_stocks:
                if each_picked not in portfolio_stocks:
                    picked_stocks.append(each_picked)
            current_money_available = UserProfile.objects.get(user=backtest_user).moneyAvailable
            num_possible_stocks = int(math.floor(current_money_available/self.stock_min))
            if num_possible_stocks > self.max_num_stocks:
                num_possible_stocks = self.max_num_stocks
            if num_possible_stocks > len(picked_stocks):
                num_possible_stocks = len(picked_stocks)
            if num_possible_stocks < 1:
                continue
            moneyAvailable_per_stock = current_money_available/num_possible_stocks

            picked_ind = 0
            count = 0
            while picked_ind < len(picked_stocks) and count < num_possible_stocks:
                stock_pick = picked_stocks[picked_ind]
                picked_ind += 1

                stock_data = backend.StockData.Instance().get_historical_stock_data(stock_pick)
                current_close = stock_data.ix[current_date].close

                num_stocks = int(math.floor(moneyAvailable_per_stock/current_close))

                if num_stocks < 0:
                    continue

                next_day_transactions.append({'stock':stock_pick, 'action':'buy',
                 'money_available':moneyAvailable_per_stock})
                count += 1

        current_invested_amount = 0.0
        portQs = UserPortfolioModel.objects.filter(username=backtest_user)
        for eachPort in portQs:
            current_invested_amount += eachPort.moneyInvested
        moneyAvailable = UserProfile.objects.get(user=backtest_user).moneyAvailable
        total = current_invested_amount+moneyAvailable
        print("****** FINAL available ***** : "+str(total))
        return total


    def calculate_stop_loss(self, data, timeperiod=3, initial=None, atr_percent=1.0):
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        open_price = data['open'].values
        #import pdb; pdb.set_trace()
        atr = talib.ATR(high, low, close, timeperiod=timeperiod)
        #chandelier_long = np.max(high[-timeperiod:]) - (atr[-1] * 1.0)
        if not self.percent_flag:
            if initial is not None:
                chandelier_long = open_price[-1] - (atr[-1] * atr_percent)
            else:
                chandelier_long = close[-1] - (atr[-1] * atr_percent)
            return chandelier_long
        if self.percent_flag:

            if initial is not None:
                return open_price[-1]-(open_price[-1]*self.stop_loss_percent/100)
            else:
                return close[-1]-(close[-1]*self.stop_loss_percent/100)


    def calculate_stop_target(self, data, timeperiod=3, initial=None, atr_percent=0.5):
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        open_price = data['open'].values
        atr = talib.ATR(high, low, close, timeperiod=timeperiod)
        if not self.percent_flag_target:
            if initial is not None:
                chandelier_long = open_price[-1] + (atr[-1] * atr_percent)
            else:
                chandelier_long = close[-1] + (atr[-1] * atr_percent)
            return chandelier_long

        if self.percent_flag_target:
            if initial is not None:
                return open_price[-1]+(open_price[-1]*self.profit_target_percent/100)
            else:
                return close[-1]+(close[-1]*self.profit_target_percent/100)

    def find_index_greater(self, target_date, date_list):
        for i in range(len(date_list)):
            temp_date = date_list[i].date()
            if temp_date >= target_date:
                start_ind = i
                break
        return start_ind

    def run_future_test(self, num_days, test_start_date=None, reset_picks = False):
        import userPortfolio.ensamble_forex as ensamble_forex
        if reset_picks:
            self.picks_dictionary = {}
        else:
            self.load_picks_dictionary()

        user_name = 'testuser'
        if len(User.objects.filter(username=user_name)):
            User.objects.get(username=user_name).delete()
        backtest_user = User.objects.create_user(user_name, 'tester@test.com', self.password)
        backtest_user.save()

        backtest_user_profile = UserProfile(user=backtest_user,
        moneyAvailable=self.starting_amount)
        backtest_user_profile.save()

        #Get the snp500 index stock and pick the starting and ending dates using the user input
        stockData = backend.StockData.Instance()
        index_data = stockData.get_historical_stock_data('^GSPC')

        start_ind = None
        end_ind = None
        if test_start_date is None:
            end_ind = len(index_data.index)-1
            start_ind = end_ind-num_days

        else:
            start_ind = self.find_index_greater(test_start_date)
            end_ind = start_ind+num_days

        all_liss = []
        next_day_transactions = []
        look_ahead = 5
        all_labels = []
        for dayInd in range(start_ind, end_ind+1-look_ahead):
            current_date = index_data.index[dayInd].date()

            all_picked_stocks = self.get_picks_date(current_date)

            future_date = index_data.index[dayInd+look_ahead].date()

            for stock_name in all_picked_stocks:
                stock_data = backend.StockData.Instance().get_historical_stock_data(stock_name)
                label_data = stock_data.ix[:future_date]

                labels = ensamble_forex.get_label(label_data.close.values, look_ahead)
                label = labels[-look_ahead-2]
                all_labels.append(label)
                print label
        return all_labels

def pick_subprocess((pick_date,)):
    print('called')
    import userPortfolio.ensamble_forex as ensamble_forex
    eble = ensamble_forex.EnsambleClassifier(num_days=1000, type='forex')
    picked_stocks = eble.pick_stocks(pick_date=pick_date)

    print('done')
    return(picked_stocks, pick_date)

# def pick_subprocess((pick_date,)):
#     print('called')
#     import userPortfolio.votingensamble as ensamble
#     eble = ensamble.EnsambleClassifier(num_days=200)
#     picked_stocks = eble.pick_stocks(pick_date=pick_date)
#
#     print('done')
#     return(picked_stocks, pick_date)


def calc_chandelier_long(data):
    timeperiod = 10
    high = data['high'].values
    low = data['low'].values
    close = data['close'].values
    atr = talib.ATR(high, low, close, timeperiod=timeperiod)
    high_22 = (close[-timeperiod:]).max()
    return high_22 - (atr[-1]*3)

#
