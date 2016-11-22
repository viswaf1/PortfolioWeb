from django.contrib import admin
from userPortfolio.models import UserProfile, UserTransactionsModel, UserPortfolioModel, AllStocksModel

admin.site.register(UserProfile)
admin.site.register(UserPortfolioModel)
admin.site.register(UserTransactionsModel)
admin.site.register(AllStocksModel)
# Register your models here.
