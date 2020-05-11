# from django.conf.urls import url
from django.urls import include, path
from userPortfolio import views

# urlpatterns = [url(r'^$', views.index, name='index'),
#         url(r'^register/$', views.register, name='register'),
#         url(r'^login/$', views.user_login, name='login'),
#         url(r'^transactions/$', views.transactions, name='transactions'),
#         #url(r'^analysis/$', views.transaction_analysis, name='analysis'),
#         url(r'^logout/$', views.user_logout, name='logout'),
#         url(r'^buy_stock/$', views.buy_stock, name='buy_stock'),
#         url(r'^sell_stock/$', views.sell_stock, name='sell_stock'),
#         url(r'^stock_plot/$', views.stock_plot, name='stock_plot')
#         ]

urlpatterns = [path('', views.index, name='index'),
        path('register/', views.register, name='register'),
        path('login/', views.user_login, name='login'),
        path('transactions/', views.transactions, name='transactions'),
        #path('analysis/', views.transaction_analysis, name='analysis'),
        path('logout/', views.user_logout, name='logout'),
        path('buy_stock/', views.buy_stock, name='buy_stock'),
        path('sell_stock/', views.sell_stock, name='sell_stock'),
        path('stock_plot/', views.stock_plot, name='stock_plot'),
        path('build_model/', views.build_model, name='build_model'),
        path('build_model/model_progress/', views.model_build_progress, name='model_progress'),
        path('build_model/model_state/', views.model_build_state, name='model_state')
        ]
