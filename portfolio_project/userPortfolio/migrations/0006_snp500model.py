# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('userPortfolio', '0005_auto_20161029_0046'),
    ]

    operations = [
        migrations.CreateModel(
            name='SNP500Model',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('stockName', models.TextField(unique=True)),
                ('market', models.TextField()),
                ('name', models.TextField()),
                ('sector', models.TextField()),
                ('industry', models.TextField()),
            ],
        ),
    ]
