## FLAR Data Source Kit

import pandas as pd
import numpy as np




## Bloomberg BQNT BQL--------------------------------------------------------------------------------


import bql
bq = bql.Service()




def get_daily_bbg_close(tickers):
    
    # Get Closing prices
    universe = tickers  # <-- Lista de strings- el string corresponde a Tickers individuales o de índices
    date_range = bq.func.range('-10Y', '0D')
    closing_prices = bq.data.px_last(dates=date_range, per='D').dropna()
    req = bql.Request(universe, {'Closing Price': closing_prices})
    res = bq.execute(req)
    # Dar formato al Dataframe
    df = res[0].df().reset_index().pivot(index='DATE', values='Closing Price', columns='ID')     # <-- parse and pivot the dataframe in one line
    return df[tickers].fillna(method='ffill').dropna()





def get_daily_bbg_ohlc(tickers):
    
    # Step 2: Use BQL to Get Data
    universe = tickers                        # <-- use a list of strings as the universe
    date_range = bq.func.range('-10Y', '0D')
    
    closing_prices = bq.data.px_last(dates=date_range, per='D').dropna()
    req = bql.Request(universe, {'Closing Price': closing_prices})
    res = bq.execute(req)
    
    open_prices = bq.data.px_open(dates=date_range, per='D').dropna()
    req_ = bql.Request(universe, {'Open Price': open_prices})
    res_ = bq.execute(req_)
    
    high_prices = bq.data.px_high(dates=date_range, per='D').dropna()
    req1 = bql.Request(universe, {'High Price': high_prices})
    res1 = bq.execute(req1)
    
    low_prices = bq.data.px_low(dates=date_range, per='D').dropna()
    req2 = bql.Request(universe, {'Low Price': high_prices})
    res2 = bq.execute(req2)
    
    # Step 3: Additional Data Munging
    close = res[0].df().reset_index().pivot(index='DATE', values='Closing Price', columns='ID')     # <-- parse and pivot the dataframes
    open_ = res_[0].df().reset_index().pivot(index='DATE', values='Open Price', columns='ID')
    high = res1[0].df().reset_index().pivot(index='DATE', values='High Price', columns='ID') 
    low = res2[0].df().reset_index().pivot(index='DATE', values='Low Price', columns='ID') 
    
    return open_, high, low , close


## REFINITIV EIKON ---------------------------------------------------------------------------------------
import requests

def get_data(list1, list2):
    # Replace this URL with the URL of your Flask service
    flask_service_url = 'http://10.130.1.37:5000/get_data'
 
    data = {
        'instruments': list1,
        'fields': list2
    }
 
    try:
        response = requests.post(flask_service_url, json=data)
 
        if response.status_code == 200:
            result_df = pd.DataFrame.from_records(response.json())
            return result_df
        else:
            return f"Request failed with status code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"Connection error: {e}"
    
## How to ask for a time series:   
##ids=['US10YT=RR']
##fields =['TR.BIDPRICE(SDate=0,EDate=-10,Frq=D).Date','TR.BIDPRICE(SDate=0,EDate=-10,Frq=D)']

