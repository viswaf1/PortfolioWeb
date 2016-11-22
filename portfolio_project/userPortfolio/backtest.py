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





#
