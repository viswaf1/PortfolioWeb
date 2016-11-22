from django.db import models
from django.contrib.auth.models import User
from django_pandas.managers import DataFrameManager



# Create your models here.

class UserProfile(models.Model):
    user = models.OneToOneField(User)
    moneyAvailable = models.FloatField(default=0.0)
    objects = DataFrameManager()

    def __unicode__(self):
        return self.user.username

class UserPortfolioModel(models.Model):
    username = models.ForeignKey(User, to_field='username')
    portfolioId = models.TextField(null=False)
    stockName = models.TextField(null=False, verbose_name='Stock Name')
    moneyInvested = models.FloatField(null=False, verbose_name='Money Invested')
    numberOfStocks = models.IntegerField(null=False, verbose_name='Number of Stocks')


class UserTransactionsModel(models.Model):
    username = models.ForeignKey(User, to_field='username')
    portfolioId = models.TextField(null=False)
    stockName = models.TextField(null=False)
    buyDate = models.DateTimeField(null=True)
    sellDate = models.DateTimeField(null=True)
    buyPrice = models.FloatField(null=True)
    sellPrice = models.FloatField(null=True)
    numberOfStocksBought = models.IntegerField(null=True)
    numberOfStocksSold = models.IntegerField(null=True)

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
