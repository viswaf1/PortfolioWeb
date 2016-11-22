# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('userPortfolio', '0004_auto_20161029_0026'),
    ]

    operations = [
        migrations.AlterField(
            model_name='usertransactionsmodel',
            name='buyDate',
            field=models.DateTimeField(null=True),
        ),
        migrations.AlterField(
            model_name='usertransactionsmodel',
            name='buyPrice',
            field=models.FloatField(null=True),
        ),
        migrations.AlterField(
            model_name='usertransactionsmodel',
            name='numberOfStocksBought',
            field=models.IntegerField(null=True),
        ),
        migrations.AlterField(
            model_name='usertransactionsmodel',
            name='numberOfStocksSold',
            field=models.IntegerField(null=True),
        ),
        migrations.AlterField(
            model_name='usertransactionsmodel',
            name='sellDate',
            field=models.DateTimeField(null=True),
        ),
        migrations.AlterField(
            model_name='usertransactionsmodel',
            name='sellPrice',
            field=models.FloatField(null=True),
        ),
    ]
