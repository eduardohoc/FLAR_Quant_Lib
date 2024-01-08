import pandas as pd
import numpy as np




# Breakout Indicators


def get_high_lows_lookback(high, low, lookback_days):
    """
    Get highs and lows in a lookbak window.
    
    Parameters
    ----------
    high : DataFrame
        High price for each ticker and date
    low : DataFrame
        Low price for each ticker and date
    lookback_days : int
        
    
    Return
    -------
    lookback_high : DataFrame
        Highest price of the lokback window
    lookback_low : DataFrame
        Lowest price of the lookback window
    """
    
    
    lookback_high = high.shift(1).rolling(window = lookback_days).max()
    lookback_low = low.shift(1).rolling(window = lookback_days).min()

    return lookback_high, lookback_low


def create_breakout_signal(close, lookback_high, lookback_low):
    """
    Generate long, short and do nothing signals.
    
    Parameters
    ----------
    close : DataFrame
        Closing prices for each ticker and date
    lookback_high : DataFrame
        Highest price of the lokback window
    lookback_low : DataFrame
        Lowest price of the lookback window
    
    Returns
    -------
    Signals : DataFrame
        long, short, and do nothing (1,0,-1) for each ticker and date
    """
    
    signals = pd.DataFrame(0, columns = close.columns, index = close.index)
    signals[lookback_high < close] = 1
    signals[lookback_low > close] = -1

    return signals



def create_breakout_signal_rates(close, lookback_high, lookback_low):
    """
    Generate long, short and do nothing signals.
    
    Parameters
    ----------
    close : DataFrame
        Closing prices for each ticker and date
    lookback_high : DataFrame
        Highest price of the lokback window
    lookback_low : DataFrame
        Lowest price of the lookback window
    
    Returns
    -------
    Signals : DataFrame
        long, short, and do nothing (1,0,-1) for each ticker and date
    """
    
    signals = pd.DataFrame(0, columns = close.columns, index = close.index)
    signals[lookback_high < close] = -1
    signals[lookback_low > close] = 1

    return signals



# MACD Indicators

def calculate_simple_moving_average(close, n):
    """
    Calculates the simple moving average.
    
    Parameters
    ----------
    Close : DataFrame
        Close prece for every ticker and date
    
    n : int
        window size for the moving average computation

    Returns
    -------
    sma : DataFrame
      Simple movig average
    
    """
    
    
    sma = close.rolling(window = n).mean()
    

    return sma



def calculate_macd_oscillator(close,n1,n2):
    """
    Calculates the moving average convergence divergences oscillator
    given a short moving average of length n1 and a long moving 
    average of length n2
    
    Parameters
    ----------
    Close : DataFrame
        Close price for every ticker and date
    
    n1: int
        short window size for the moving average computation
    n2: int
        long window size for the moving average computation    

    Returns
    -------
    macd : DataFrame
      MACD Oscillator
    
    """
    

    assert n1 < n2
    
    macd = calculate_simple_moving_average(close, n1) - calculate_simple_moving_average(close, n2)
    

    return macd

def create_macd_signal(close,n1,n2):
    """
    Create a momentum-based signal based on the MACD crossover principle:
    Buy signal when MACD cross above zero, sell signal when MACD croses 
    below zero.
    
    Parameters
    ----------
    Close : DataFrame
        Close price for every ticker and date
    
    n1: int
        short window size for the moving average computation
    n2: int
        long window size for the moving average computation    

    Returns
    -------
    Signals : DataFrame
      Buy (1) Sell (-1) or do nothing signal (0)
    """
    #TODO: Implement function
    # Calculate macd and get the signs of the values
    macd = calculate_macd_oscillator(close,n1,n2)
    macd_sign = np.sign(macd)
    #Create a copy shifted
    macd_shifted_sign = macd_sign.shift(1,axis=0)
    #Multiply by the sign by boolean to creates signals 
    signals = macd_sign*(macd_sign !=macd_shifted_sign)

    return signals


# Bollinger Band Indicators


def calculate_simple_moving_sample_stdev(close, n):
    """
    Calculates the simple moving standard deviation.
    
    Parameters
    ----------
    Close : DataFrame
        Close prece for every ticker and date
    
    n : int
        window size for the moving average computation

    Returns
    -------
    sma : DataFrame
      Simple movig standard deviation
    
    """
    #TODO: Implement function
    
    smsd = close.rolling(window = n).std()
    

    return smsd


def create_bollinger_band_signal(close, n):
    
    """
    Create a meanreverting-based signal based on the upper and lower 
    bands of the Bollinger bands. Geenerate a buy sigal when the price 
    is bellow the lower band and a sell signal when the price is above
    the uper band.

    
    Parameters
    ----------
    Close : DataFrame
    Close price for every ticker and date
    
    n: int
    window size for the moving average and standard deviation computation
    
    Returns
    -------
    Signals : DataFrame
    Buy (1) Sell (-1) or do nothing signal (0)
    """
    sma = calculate_simple_moving_average(close,n)
    stdev = calculate_simple_moving_sample_stdev(close, n)
    upper = sma + 2*stdev
    lower = sma - 2*stdev
    
    sell = close > upper
    buy = close < lower
    signal = 1*buy - 1* sell
    return signal
    

### RSI Indicator 

def compute_rsi(prices, n=14):
    
    """
    Calculates the Relative strenght index (RSI) of a series of prices.
    
    Parameters
    ----------
    prices : Series
        Close prece Series
    n: Number of days to compute the RSI
    Returns
    -------
    rsi : Series
      Relative strenght Index series
     """  
    
    
    deltas = np.diff(prices)
    seed = deltas[:n + 1]
    up = seed[seed >= 0].sum() / n
    down = -seed[seed < 0].sum() / n
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100. / (1. + rs)

    for i in range(n, len(prices)):
        delta = deltas[i - 1]

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (n - 1) + upval) / n
        down = (down * (n - 1) + downval) / n
        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi
 
 
def create_rsi_signal(close, n=14,up=60,down=40):
    """
    Generate long, short and do nothing signals based on Relative strengnht index overbught/oversold state.
    
    Parameters
    ----------
    close : DataFrame
        Closing prices for each ticker and date
    low : DataFrame
        low prices for each ticker and date    
    
    Returns
    -------
    Signals : DataFrame
        long, short, and do nothing (1,0,-1) for each ticker and date
    """
    rsi =  pd.DataFrame(0, columns = close.columns, index = close.index)
    for column in close.columns:
        rsi[column] = compute_rsi(close[column],n)
        
    signals = pd.DataFrame(0, columns = close.columns, index = close.index)
    signals[rsi>up] = -1
    signals[rsi<down]= 1
    
    return signals

# Candle pattern Indicators 

def upper_shadow(high,close,open_):
    
    """
    Calculates the length of the candle's upper shadow.
    
    Parameters
    ----------
    high : DataFrame
        High prices for each ticker and date
    close : DataFrame
        Closing prices for each ticker and date
    open_ : DataFrame
        Open prices for each ticker and date    
    
    Returns
    -------
    upper_shadiw : DataFrame
        Length of upper shadow for each ticker and date
    """

    return high - np.maximum(close, open_)

def lower_shadow(close,open_,low):
    """
    Calculates the length of the candle's lower shadow.
    
    Parameters
    ----------
    close : DataFrame
        Closing prices for each ticker and date
    open_ : DataFrame
        Open prices for each ticker and date  
    low : DataFrame
        Lowprices for each ticker and date
    
    Returns
    -------
    lower_shadow : DataFrame
        Length of upper shadow for each ticker and date
    """

    return np.minimum(close, open_) - low

def body_size (open_,close):  
    """
    Calculates the length of the candle's body.
    
    Parameters
    ----------
    open_ : DataFrame
        Open prices for each ticker and date   
    close : DataFrame
        Closing prices for each ticker and date   
    
    Returns
    -------
    body : DataFrame
        Length of candle's body for each ticker and date
    """
    return abs(open_-close)
    
def create_reversal_signal(high,open_,close,low, l1=2.5,l2=0.3):
    """
    Generate long, short and do nothing signals based on shooting star and hammer candlestick formations.
    
    Parámetross
    ----------
    high : DataFrame
        High prices for each ticker and date
    open_ : DataFrame
        Open prices for each ticker and date
    close : DataFrame
        Closing prices for each ticker and date
    low : DataFrame
        low prices for each ticker and date    
    
    Returns
    -------
    Signals : DataFrame
        long, short, and do nothing (1,0,-1) for each ticker and date
    """
    upper_s = upper_shadow (high,close,open_)
    lower_s = lower_shadow(close,open_,low)
    body = body_size (open_,close)
    
    ratio1 = upper_s/ body
    ratio2 = lower_s/ body 
    
    signals = pd.DataFrame(0, columns = close.columns, index = close.index)
    signals[(ratio1 >l1) & (lower_s< l2*upper_s)] = -1
    signals[(ratio2 >l1) & (upper_s< l2*lower_s)] = 1
    

    return signals   
    
    