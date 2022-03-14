# Long pipeline
# coding: utf-8
import os
import yfinance as yf
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta, date, time
import pytz
import pandas as pd
import pandas_ta as ta
import pandas_market_calendars as mcal
import numpy as np
import smtplib
from binance.client import Client
import vectorbt as vbt
import copy
import  time as  tid

def binance_pubsub(criteria1, criteria2):
    tickers = ['ETHUSDT','BTCUSDT','LTCUSDT','XLMUSDT','EOSUSDT','XRPUSDT','XTZUSDT','OMGUSDT',
                             'ZRXUSDT','BNBUSDT','LINKUSDT','XMRUSDT','IOTAUSDT','ADAUSDT','DOTUSDT','DOGEUSDT']
    data = {}
    df_recalc = {}
    prices_rec = pd.DataFrame()
    windows = np.arange(2, 30)
    
    client = Client(os.environ['IPA_k'],os.environ['IPA_s'])
    hoy = dt.utcnow().date()
    d1_now = hoy.strftime('%b %d,%Y')
    d1 = (hoy - timedelta(days=1)).strftime('%b %d,%Y')

    # Prepare calendar of valid days
    nyse = mcal.get_calendar('NYSE')
    holidays = nyse.holidays()
    time = nyse.valid_days(start_date='2017-01-01', end_date='2050-01-01')
    time = time.to_frame(index=False, name='Date')
    time['day_status'] = 'valid'

    # Prepare dataframes with needed columns
    for ticker in tickers:
        klines = client.get_historical_klines(ticker, client.KLINE_INTERVAL_1DAY, "27 Sep, 2017", d1_now)
        df = pd.DataFrame.from_records(klines)
        df = df.rename(columns={0:"Date", 1:"open", 2:"high", 3:"low", 4:"close",5:"volume"})
        df = df[['Date','open','high','low','close','volume']]
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df['Date'] = df['Date'].dt.tz_localize('UTC')
        df = time.merge(df, on='Date', how='inner')
        df["open"] = pd.to_numeric(df["open"], downcast="float")
        df["high"] = pd.to_numeric(df["high"], downcast="float")
        df["low"] = pd.to_numeric(df["low"], downcast="float")
        df["close"] = pd.to_numeric(df["close"], downcast="float")
        df["volume"] = pd.to_numeric(df["volume"], downcast="float")    
        data[ticker] = df


    context_stocks = tickers 
    body = ''
    body2 = ''

    chg = False

    # Calculate total balance
    balance_usdt = client.get_asset_balance(asset='USDT')
    total_amount = float(balance_usdt['free'])
    for stock in context_stocks:
        coin_lab = stock.partition("USDT")[0]
        balance_quote = client.get_asset_balance(asset=coin_lab)
        avg_price = client.get_avg_price(symbol=coin_lab+'USDT')
        coin_in_usdt = (float(balance_quote['free'])+float(balance_quote['locked']))*float(avg_price['price'])
        total_amount = total_amount + coin_in_usdt
    print(total_amount)
    
    # Run strategy
    for stock in context_stocks:
            print(stock)

            #calculate SMAs
            sma = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
            for time in sma:
                data[stock][str('sma_'+str(time))] = ta.sma(data[stock]['close'], length=time)

            #print(rsi)
            rsi = [4,5,7,10]
            for time in rsi:
                data[stock][str('rsi'+str(time))] = ta.rsi(data[stock]['close'], length=time)

            # volatility
            data[stock]['natr'] = ta.natr(data[stock]['high'],data[stock]['low'],data[stock]['close'], length=10)
            data[stock]['sma_vol'] = ta.sma(data[stock]['natr'], length=10) 

            # volume indicators
            data[stock]['obv'] = ta.obv(data[stock]['close'],data[stock]['volume'])
            data[stock]['adosc'] = ta.adosc(data[stock]['high'], data[stock]['low'], data[stock]['close'], data[stock]['volume'],fast=7,slow=20)

            # STRAT
            data[stock].insert(0, 'index_int', range(0, len(data[stock])))
            data[stock] = data[stock].set_index('index_int')

            # Use custom fast and slow
            df_recalc[stock] = copy.deepcopy(data[stock])
            prices_rec[stock] = df_recalc[stock]['close']
            fast_ma, slow_ma = vbt.MA.run_combs(prices_rec[stock], window=windows, r=2, short_names=['fast', 'slow'])
            entries = fast_ma.ma_above(slow_ma, crossover=True)
            exits = fast_ma.ma_below(slow_ma, crossover=True)
            portfolio_kwargs = dict(init_cash=2000, size=np.inf, fees=0.002, freq='1D')
            portfolio = vbt.Portfolio.from_signals(prices_rec[stock], entries, exits, direction='longonly', **portfolio_kwargs)
            df_returns = pd.DataFrame(portfolio.total_return())
            maxi = df_returns[['total_return']]==portfolio.total_return().max()
            maxi = maxi.loc[maxi['total_return']==True]
            fast = maxi.index[0][0]
            slow = maxi.index[0][1]
            print(fast, slow)
            data[stock]['ha1_cond'] = np.where((data[stock][str('sma_'+str(fast))] > data[stock][str('sma_'+str(slow))]), 1, 0) # (data[stock]['sma_vol'] < 20
            data[stock]['ha1_reac'] = np.where((data[stock]['rsi5'] > 25) & (data[stock]['rsi5'] < 80), 1,0)
            data[stock]['entry_price'] = np.nan
            data[stock]['limit_top'] = np.nan
            data[stock]['entry'] = np.nan
            data[stock]['exit'] = np.nan
            entry_price1=0
            entry_price2=0
            cond = 0

            position = 0
            entry_date = 0

            for index, row in data[stock].iterrows():
                if str(data[stock].loc[index, 'Date'])=='2021-02-22 00:00:00+00:00':
                    position = 0
                elif (position == 0 and data[stock].loc[index, 'ha1_cond'] == 1 and (data[stock].loc[index, 'ha1_reac'] == 1) and (data[stock].loc[index, 'close'] > data[stock].loc[index-2, 'close'])):
                    data[stock].loc[index, 'entry'] = 1
                    entry_price1 = data[stock].loc[index, 'close']
                    entry_price2 = data[stock].loc[index, 'close']
                    entry_date = 0
                    position = 1
                    cond = 1
                elif (position == 1 and (entry_price2*1.5 < data[stock].loc[index, 'close'])):
                    # Ici rajout du rachat et variables liés en même temps permet d'atteindre un très haut rendement
                    data[stock].loc[index, 'exit'] = 2
                    position = 0
                    cond = 2
                elif (position == 1 and (entry_price2*1.5 < data[stock].loc[index, 'high'])):
                    # Trigger for limit function
                    data[stock].loc[index, 'limit_top'] = 1
                    position = 0
                    cond = 3
                elif (position==1 and data[stock].loc[index, 'rsi7'] > 65 and (entry_price1 < data[stock].loc[index, 'close'])) or (position == 1 and (entry_price2*1.05 < data[stock].loc[index, 'close'])):
                    entry_price2 = data[stock].loc[index, 'close']
                    entry_date = 0
                    cond = 4
                elif (position == 1 and data[stock].loc[index, 'ha1_cond'] == 0) or entry_date >= 8 or (position == 1 and (entry_price1*0.85 > data[stock].loc[index, 'close'])):
                    # Check if I can refine this
                    data[stock].loc[index, 'exit'] = 1
                    position = 0
                    cond = 5
                elif position == 1:
                    entry_date = entry_date + 1
                    cond = 6
            body2 = body2 + str("\n"+stock + " : " + str(position) + "; day #" + str(entry_date) +"; cond #" + str(cond))

            # Info and buy/sell amounts
            coin_lab = stock.partition("USDT")[0]
            info = client.get_symbol_info(stock)
            balance_usdt = client.get_asset_balance(asset='USDT')
            balance_quote = client.get_asset_balance(asset=coin_lab)
            avg_price = client.get_avg_price(symbol=coin_lab+'USDT')
            rounding = len(info['filters'][2]['stepSize'].rstrip('0').partition(".")[2])
            buy_amount = (total_amount/10)/float(avg_price['price'])
            buy_amount = round(buy_amount, rounding)
            limit_sell_amount = round(buy_amount*0.995, rounding)
            #oco_amount_renew = round(buy_amount*0.998, rounding)
            if float(balance_quote['free'])+float(balance_quote['locked'])>0 and str(round(float(balance_quote['free'])+float(balance_quote['locked']), rounding))[-1]!='0': 
                sell_amount = float(str(round(float(balance_quote['free'])+float(balance_quote['locked']), rounding))[:-1] + str(int(str(round(float(balance_quote['free'])+float(balance_quote['locked']), rounding))[-1])-1))
            elif float(balance_quote['free'])+float(balance_quote['locked'])>0 and str(round(float(balance_quote['free'])+float(balance_quote['locked']), rounding))[-1]=='0':
                sell_amount = float(str(round(float(balance_quote['free'])+float(balance_quote['locked']), rounding))[:-1])-1
            elif float(balance_quote['free'])+float(balance_quote['locked'])==0:
                sell_amount = 0
            
            # System to spend the remaining part of the USDT if below total balance/6
            if round(float(balance_usdt['free'])+float(balance_usdt['locked']),rounding) >= (total_amount/7):
                buy = 1
            elif round(float(balance_usdt['free'])+float(balance_usdt['locked']),rounding) <= (total_amount/7) and round(float(balance_usdt['free'])+float(balance_usdt['locked']),rounding) >= 50:
                buy = 1
                buy_amount = (float(balance_usdt['free'])*0.99)/float(avg_price['price'])
                buy_amount = round(buy_amount, rounding)
            else:
                buy = 0
            sell = sell_amount*float(avg_price['price']) > 10
            
            # Important for stop limit order
            rounding_2 = len(info['filters'][0]['tickSize'].rstrip('0').partition(".")[2])
            limit_p = str(round(float(avg_price['price'])*0.49, rounding_2))
            stop_p = str(round(float(avg_price['price'])*0.50, rounding_2))
            top_p = str(round(float(avg_price['price'])*1.5, rounding_2))

            # Adapt limit sell for each new day
            orders = client.get_all_orders(symbol=stock)
            if position==1:
                for order in orders:
                    if order['type'] == 'LIMIT' and order['status'] == 'NEW':
                        result = client.cancel_order(symbol=stock, orderId=order['orderId'])
                if sell: 
                    order = client.order_limit_sell(symbol=stock, quantity=sell_amount, price=top_p)

            # Final triggers
            if data[stock]['entry'].iloc[-1]==1:
                body = body + 'Entry %s! - ' % stock
                if buy:
                    order = client.order_market_buy(symbol = coin_lab+'USDT', quantity = buy_amount)
                    tid.sleep(100)
                    order = client.order_limit_sell(symbol= coin_lab+'USDT', quantity = limit_sell_amount, price=top_p)
                chg = True
            elif data[stock]['exit'].iloc[-1]==1 or data[stock]['exit'].iloc[-1]==2:
                if data[stock]['exit'].iloc[-1]==1:
                    body = body + 'Exit %s! - ' % stock
                if data[stock]['exit'].iloc[-1]==2:
                    body = body + 'Exit win %s! - ' % stock
                if sell:
                    for order in orders:
                        if order['type'] == 'LIMIT' and order['status'] == 'NEW':
                            result = client.cancel_order(symbol=stock, orderId=order['orderId'])
                    tid.sleep(5)
                    order = client.order_market_sell(symbol=coin_lab+'USDT', quantity=sell_amount)
                chg = True
            elif data[stock]['limit_top'].iloc[-1]==1:
                body = body + 'Exit limit top %s! - ' % stock
                chg = True
            else:
                body = body

    body3 = body + body2 + "\nend date: "+str(data[stock].iloc[-1]['Date'].date())

    # SEND FUNCTIONS #
    def send_email(sbjt, msg):
        fromaddr = 'X'
        toaddrs = 'X'
        bodytext = 'From: %s\nTo: %s\nSubject: %s\n\n%s' %(fromaddr, toaddrs, sbjt, msg)

        # The actual mail sent
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.starttls()
        server.login(os.environ['email_p'],os.environ['pass_p'])
        server.sendmail(fromaddr, toaddrs, bodytext)
        server.quit()

    if chg == True:
        print('sending email...')
        send_email('Long positions', body3)

    elif chg == False:
        send_email('No long positions', body3)

    print("SUCCESS!")