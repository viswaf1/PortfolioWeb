from django.contrib import admin
from userPortfolio.models import UserProfile, UserTransactionsModel, UserPortfolioModel, AllStocksModel, USDForexModel

admin.site.register(UserProfile)
admin.site.register(UserPortfolioModel)
admin.site.register(UserTransactionsModel)
admin.site.register(AllStocksModel)
admin.site.register(USDForexModel)
# Register your models here.
