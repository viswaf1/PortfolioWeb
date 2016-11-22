# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models
from django.conf import settings


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('userPortfolio', '0003_auto_20161027_0347'),
    ]

    operations = [
        migrations.CreateModel(
            name='UserPortfolioModel',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('portfolioId', models.TextField()),
                ('stockName', models.TextField(verbose_name=b'Stock Name')),
                ('moneyInvested', models.FloatField(verbose_name=b'Money Invested')),
                ('numberOfStocks', models.IntegerField(verbose_name=b'Number of Stocks')),
                ('username', models.ForeignKey(to=settings.AUTH_USER_MODEL, to_field=b'username')),
            ],
        ),
        migrations.CreateModel(
            name='UserTransactionsModel',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('portfolioId', models.TextField()),
                ('stockName', models.TextField()),
                ('buyDate', models.DateTimeField()),
                ('sellDate', models.DateTimeField()),
                ('buyPrice', models.FloatField()),
                ('sellPrice', models.FloatField()),
                ('numberOfStocksBought', models.IntegerField()),
                ('numberOfStocksSold', models.IntegerField()),
                ('username', models.ForeignKey(to=settings.AUTH_USER_MODEL, to_field=b'username')),
            ],
        ),
        migrations.RenameModel(
            old_name='AllStocks',
            new_name='AllStocksModel',
        ),
        migrations.RemoveField(
            model_name='userportfolio',
            name='username',
        ),
        migrations.RemoveField(
            model_name='usertransactions',
            name='username',
        ),
        migrations.DeleteModel(
            name='userPortfolio',
        ),
        migrations.DeleteModel(
            name='userTransactions',
        ),
    ]
