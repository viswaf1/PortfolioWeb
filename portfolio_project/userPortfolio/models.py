from django.db import models
from django.contrib.auth.models import User
from django_pandas.managers import DataFrameManager



# Create your models here.

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    moneyAvailable = models.FloatField(default=0.0)
    #objects = DataFrameManager()

    def __unicode__(self):
        return self.user.username

class UserPortfolioModel(models.Model):
    username = models.ForeignKey(User, to_field='username', on_delete=models.CASCADE)
    portfolioId = models.TextField(null=False)
    stockName = models.TextField(null=False, verbose_name='Stock Name')
    moneyInvested = models.FloatField(null=False, verbose_name='Money Invested')
    numberOfStocks = models.IntegerField(null=False, verbose_name='Number of Stocks')
    stopLoss = models.FloatField(null=True, verbose_name='Stop Loss')
    minStopLoss = models.FloatField(null=True, verbose_name='Min Stop Loss')
    stopTarget = models.FloatField(null=True, verbose_name='Target')


class UserTransactionsModel(models.Model):
    username = models.ForeignKey(User, to_field='username', on_delete=models.CASCADE)
    portfolioId = models.TextField(null=False)
    stockName = models.TextField(null=False)
    buyDate = models.DateTimeField(null=True)
    sellDate = models.DateTimeField(null=True)
    buyPrice = models.FloatField(null=True)
    sellPrice = models.FloatField(null=True)
    numberOfStocksBought = models.IntegerField(null=True)
    numberOfStocksSold = models.IntegerField(null=True)
    returns = models.FloatField(null=True)
    reason = models.TextField(null=True)

class AllStocksModel(models.Model):
    stockName = models.TextField(null=False, unique=True)
    market = models.TextField()
    name = models.TextField()
    sector = models.TextField()
    industry = models.TextField()

class SNP500Model(models.Model):
    stockName = models.TextField(null=False, unique=True)
    market = models.TextField()
    name = models.TextField()
    sector = models.TextField()
    industry = models.TextField()

class USDForexModel(models.Model):
    stockName = models.TextField(null=False, unique=True)
    


class BacktestDataModel(models.Model):
    backtest_id = models.TextField()
    username = models.ForeignKey(User, to_field='username', on_delete=models.CASCADE)
    date = models.DateTimeField(null=False)
    buySignalData = models.TextField(null=False)
    sellSignalData = models.TextField(null=False)



#
