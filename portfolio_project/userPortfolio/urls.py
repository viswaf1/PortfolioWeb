from django.conf.urls import patterns, url
from userPortfolio import views

urlpatterns = patterns('',
        url(r'^$', views.index, name='index'),
        url(r'^register/$', views.register, name='register'),
        url(r'^login/$', views.user_login, name='login'),
        url(r'^transactions/$', views.transactions, name='transactions'),
        url(r'^logout/$', views.user_logout, name='logout'),
        url(r'^buy_stock/$', views.buy_stock, name='buy_stock'),
        url(r'^sell_stock/$', views.sell_stock, name='sell_stock'),
        url(r'^stock_plot/$', views.stock_plot, name='stock_plot')
        )
