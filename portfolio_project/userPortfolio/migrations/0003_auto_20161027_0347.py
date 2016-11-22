# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('userPortfolio', '0002_userportfolio_usertransactions'),
    ]

    operations = [
        migrations.CreateModel(
            name='AllStocks',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('stockName', models.TextField(unique=True)),
                ('market', models.TextField()),
                ('name', models.TextField()),
                ('sector', models.TextField()),
                ('industry', models.TextField()),
            ],
        ),
        migrations.AlterField(
            model_name='userportfolio',
            name='moneyInvested',
            field=models.FloatField(verbose_name=b'Money Invested'),
        ),
        migrations.AlterField(
            model_name='userportfolio',
            name='numberOfStocks',
            field=models.IntegerField(verbose_name=b'Number of Stocks'),
        ),
        migrations.AlterField(
            model_name='userportfolio',
            name='stockName',
            field=models.TextField(verbose_name=b'Stock Name'),
        ),
    ]
