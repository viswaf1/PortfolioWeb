# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models
from django.conf import settings


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('userPortfolio', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='userPortfolio',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('portfolioId', models.TextField()),
                ('stockName', models.TextField()),
                ('moneyInvested', models.FloatField()),
                ('numberOfStocks', models.IntegerField()),
                ('username', models.ForeignKey(to=settings.AUTH_USER_MODEL, to_field=b'username')),
            ],
        ),
        migrations.CreateModel(
            name='userTransactions',
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
    ]
