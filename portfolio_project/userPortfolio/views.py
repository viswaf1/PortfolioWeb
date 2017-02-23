from django.shortcuts import render, render_to_response
from userPortfolio.forms.forms import UserForm, UserProfileForm
from django.http import HttpResponseRedirect, HttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from userPortfolio.models import UserTransactionsModel, UserPortfolioModel, AllStocksModel
import userPortfolio.backend as backend
import datetime




@login_required
def index(request):

    qs = UserPortfolioModel.objects.filter(username=request.user.username)
    port_table = render_portfolio_table(qs)

    context_dict = {'username_apos' : (request.user.username.title()+"'s"), \
    'user' : (request.user.username).title(), \
    'portfolio_table' : port_table}
    return render(request, 'userPortfolio/index.html' ,context_dict)


@login_required
def transactions(request):
    if request.method == 'GET':
        stockName = request.GET.get('stock', '')
        html_table = render_transaction_table(stockName, request.user)
    else:
        html_table = render_transaction_table('', request.user)

    script, div = backend.render_transaction_sales(request.user)

    context_dict = {'username_apos' : (request.user.username.title()+"'s"), \
    'user' : (request.user.username).title(), 'transaction_table' : html_table,\
     'script' : script , 'div' : div}

    return render(request, 'userPortfolio/transactions.html' ,context_dict)

# Create your views here.
def register(request):
    # A boolean value for telling the template whether the registration was successful.
    # Set to False initially. Code changes value to True when registration succeeds.
    registered = False

    # If it's a HTTP POST, we're interested in processing form data.
    if request.method == 'POST':
        user_form = UserForm(data=request.POST)
        profile_form = UserProfileForm(data=request.POST)

        if user_form.is_valid() and profile_form.is_valid():
            # Save the user's form data to the database.
            user = user_form.save()

            # Now we hash the password with the set_password method.
            # Once hashed, we can update the user object.
            user.set_password(user.password)
            user.save()

            profile = profile_form.save(commit=False)
            profile.user = user

            # Now we save the UserProfile model instance.
            profile.save()
            # Update our variable to tell the template registration was successful.
            registered = True

        else:
            print user_form.errors, profile_form.errors
    else:
        user_form = UserForm()
        profile_form = UserProfileForm()

    # Render the template depending on the context.
    return render(request,
            'userPortfolio/register.html',
            {'user_form': user_form, 'profile_form': profile_form, 'registered': registered} )

def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(username=username, password=password)
        if user:
            if user.is_active:
                login(request, user)
                return HttpResponseRedirect('/userPortfolio/')
            else:
                return HttpResponse("Your Rango account is disabled.")
        else:
            print "Invalid login details: {0}, {1}".format(username, password)
            return HttpResponse("Invalid login details supplied.")

    else:
        return render(request, 'userPortfolio/login.html', {})

@login_required
def user_logout(request):
    # Since we know the user is logged in, we can now just log them out.
    logout(request)
    # Take the user back to the homepage.
    return HttpResponseRedirect('/userPortfolio/')

@login_required
def stock_plot(request):

    if request.method == 'GET':
        stockName = request.GET.get('stock', '')
    else:
        return render( request, 'userPortfolio/stock_plot.html', {'script' : '' , 'div' : ''} )

    (script, div) = backend.render_stock_data(stockName)

    #Feed them to the Django template.
    return render( request, 'userPortfolio/stock_plot.html', {'script' : script , 'div' : div} )

@login_required
def buy_stock(request):
    return buy_sell_stock(request, "buy_stock")

@login_required
def sell_stock(request):
    return buy_sell_stock(request, "sell_stock")

def buy_sell_stock(request, type):
    post_error = False
    post_success = False
    post_message = ""
    error_message = ""
    context_dict = {'username_apos' : (request.user.username.title()+"'s"), \
    'user' : (request.user.username).title(), 'username_upper' : (request.user.username).upper() \
    , 'type' : type, 'error_message' : error_message, 'post_error' : post_error, 'post_success' : post_success, \
    'post_message' : post_message}

    if request.method == 'POST':
        stockName = request.POST.get("stockName")
        stockNameQs = AllStocksModel.objects.filter(stockName = stockName)
        if not (len(stockNameQs) == 1):
            error_message = "Bad Stock Name"
            post_error = True
        try:
            stockPrice = float(request.POST.get("stockPrice"))
        except:# ValueError, TypeError:
            error_message = "Wrong Buy Price"
            post_error = True
        try:
            numOfStocks = int(request.POST.get("numberOfStocks"))
        except:# ValueError, TypeError:
            error_message = "Wrong Number of Stocks"
            post_error = True
        try:
            transactionDate = datetime.datetime.strptime(request.POST.get("transactionDate"), "%m/%d/%Y").date()
        except:
            error_message = "Wrong Buy Date"
            post_error = True

        if(post_error):
            context_dict['post_error'] = post_error
            context_dict['error_message'] = error_message
        else:
            if type == "buy_stock":
                msg = stockName+" Successfully added to Portfolio"
                backend.buy_stock(request.user, stockName, stockPrice, numOfStocks,\
                transactionDate)
            else:
                portQs = UserPortfolioModel.objects.filter(username = request.user, stockName = stockName)
                if len(portQs) < 1:
                    context_dict['post_success'] = False
                    msg = "Error deleting stock. "+stockName+" Not present in Portfolio"
                else:
                    ret =  backend.sell_stock(request.user, stockName, stockPrice, numOfStocks,\
                    transactionDate)
                    if(ret):
                        msg = "Successfully deleted "+stockName+" from Portfolio"
                    else:
                        msg = "Error selling the stock: "+stockName
            context_dict['post_success'] = True
            context_dict['post_message'] = msg


    return render(request, 'userPortfolio/buy_stock.html', context_dict)


# Tables for the views
def render_portfolio_table(qs):
    fields = ('stockName', 'moneyInvested', 'numberOfStocks')#, 'currentOutcome')
    fields_names = {'stockName' : 'Stock Name', 'moneyInvested' : '$ Invested',\
     'numberOfStocks' : '# Stocks', 'currentOutcome' : 'Current Outcome'}
    attrs = {'align' : 'center'}
    tableHtml = '''<div class="table-container">
    <table class="pure-table" id="portfolio_table">
        <thead>
            <tr>\n'''

    for k in fields:
        tableHtml = tableHtml + '<th'
        for a in attrs.keys():
            tableHtml = tableHtml + ' ' + a + ' = "' + attrs[a] + '" '

        tableHtml = tableHtml +' >' + fields_names[k] + '</th>\n'

    tableHtml = tableHtml+'''           </tr>
        </thead>
    <tbody>'''

    for eachRow in qs:
        tableHtml = tableHtml + '<tr onclick="tablerow_onclick(this)" stock_name = "'+ str(getattr(eachRow, 'stockName')) +'" >\n'
        for k in fields:
            tableHtml = tableHtml + '<td'
            for a in attrs.keys():
                tableHtml = tableHtml + ' ' + a + ' = "' + attrs[a] + '" '
            tableHtml = tableHtml + '>' + str(getattr(eachRow, k)) + '</td>\n'
        tableHtml = tableHtml + '</tr>\n'

        #current_price = get_current_price(eachRow.stockName)

    tableHtml = tableHtml+ '''      </tbody>
        </table>
    </div>'''

    return tableHtml

def render_transaction_table(stockName, user):
    qs = []
    if(stockName == ''):
        qs = UserTransactionsModel.objects.filter(username=user)
    else:
        port_id_qs = UserPortfolioModel.objects.filter(stockName = stockName)
        if len(port_id_qs) == 1:
            port_id = port_id_qs[0].portfolioId
            qs = UserTransactionsModel.objects.filter(portfolioId = port_id)

    fields = ('stockName', 'buyDate', 'sellDate', 'buyPrice', 'sellPrice', \
    'numberOfStocksBought', 'numberOfStocksSold', 'returns', 'reason')
    fields_names = {'stockName' : 'Stock Name', 'buyDate' : 'Buy Date', 'sellDate' : 'Sell Date',\
    'buyPrice' : 'Buy Price', 'sellPrice' : 'Sell Price', 'numberOfStocksBought' : '# Bought', \
    'numberOfStocksSold' : '# Sold', 'returns' : 'Returns', 'reason' : 'Reason'}
    attrs = {'align' : 'center'}

    tableHtml = '''<div class="table-container">
    <table class="pure-table">
        <thead>
            <tr>\n'''

    for k in fields:
        tableHtml = tableHtml + '<th'
        for a in attrs.keys():
            tableHtml = tableHtml + ' ' + a + ' = "' + attrs[a] + '" '

        tableHtml = tableHtml +' >' + fields_names[k] + '</th>\n'

    tableHtml = tableHtml+'''           </tr>
        </thead>
    <tbody>'''

    for eachRow in qs:
        tableHtml = tableHtml + '<tr>\n'
        for k in fields:
            tableHtml = tableHtml + '<td'
            for a in attrs.keys():
                tableHtml = tableHtml + ' ' + a + ' = "' + attrs[a] + '" '
            value = '-'
            if getattr(eachRow, k):
                if k == 'buyDate' or k == 'sellDate':
                    value = str(getattr(eachRow, k).strftime("%d/%m/%y"))
                else:
                    value = str(getattr(eachRow, k))

            tableHtml = tableHtml + '>' + value + '</td>\n'
        tableHtml = tableHtml + '</tr>\n'

        #current_price = get_current_price(eachRow.stockName)

    tableHtml = tableHtml+ '''      </tbody>
        </table>
    </div>'''

    return tableHtml
