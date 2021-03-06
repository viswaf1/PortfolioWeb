# Generated by Django 2.2.3 on 2019-07-14 18:15

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='AllStocksModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('stockName', models.TextField(unique=True)),
                ('market', models.TextField()),
                ('name', models.TextField()),
                ('sector', models.TextField()),
                ('industry', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='SNP500Model',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('stockName', models.TextField(unique=True)),
                ('market', models.TextField()),
                ('name', models.TextField()),
                ('sector', models.TextField()),
                ('industry', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='USDForexModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('stockName', models.TextField(unique=True)),
            ],
        ),
        migrations.CreateModel(
            name='UserTransactionsModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('portfolioId', models.TextField()),
                ('stockName', models.TextField()),
                ('buyDate', models.DateTimeField(null=True)),
                ('sellDate', models.DateTimeField(null=True)),
                ('buyPrice', models.FloatField(null=True)),
                ('sellPrice', models.FloatField(null=True)),
                ('numberOfStocksBought', models.IntegerField(null=True)),
                ('numberOfStocksSold', models.IntegerField(null=True)),
                ('returns', models.FloatField(null=True)),
                ('reason', models.TextField(null=True)),
                ('username', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL, to_field='username')),
            ],
        ),
        migrations.CreateModel(
            name='UserProfile',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('moneyAvailable', models.FloatField(default=0.0)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='UserPortfolioModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('portfolioId', models.TextField()),
                ('stockName', models.TextField(verbose_name='Stock Name')),
                ('moneyInvested', models.FloatField(verbose_name='Money Invested')),
                ('numberOfStocks', models.IntegerField(verbose_name='Number of Stocks')),
                ('stopLoss', models.FloatField(null=True, verbose_name='Stop Loss')),
                ('minStopLoss', models.FloatField(null=True, verbose_name='Min Stop Loss')),
                ('stopTarget', models.FloatField(null=True, verbose_name='Target')),
                ('username', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL, to_field='username')),
            ],
        ),
        migrations.CreateModel(
            name='BacktestDataModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('backtest_id', models.TextField()),
                ('date', models.DateTimeField()),
                ('buySignalData', models.TextField()),
                ('sellSignalData', models.TextField()),
                ('username', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL, to_field='username')),
            ],
        ),
    ]
