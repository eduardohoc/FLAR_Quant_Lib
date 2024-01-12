import pandas as pd
import numpy as np
import math


#### Retuns , compounding, annualize 

def rolling_sum(series,window):
    
    """
    Computes rolling sum based on a set window 

    Parameters
    ----------
    series: Pandas Series
    
    window : int
    window to compute the  rolling sum
    """
    r = series.rolling(window=window)
    rs=r.sum().shift(1)
   
    return rs


# Resample prices (Default Monthly)
def resample_prices(close_prices, freq='M'):
    """
    From a daily DataFrame , get a monthly DataFrame with the last closing value in the last date.
    
    Parameters
    ----------
    close_prices : DataFrame/Serie
        Daily close prices for each ticker and date
    freq : str
        Sampling frequency
        PTo get valid frequecies consult:
        see http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    
    Returns
    -------
    prices_resampled : DataFrame
        Closeing prices in the monthly frequecy
    """
   
    
    prices_resampled  = close_prices.resample(freq).last()
    
    return prices_resampled




def get_return(Prices):

    """
    Computes return based on a Series or DataFrame of prices

    Parameters
    ----------
    prices: Pandas Series
    
    """
    
    
    return Prices.pct_change()


def compound(r):
    """
    returns the result of compounding the set of returns in r
    """
    return np.expm1(np.log1p(r).sum())

                         
def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    return r.std()*(periods_per_year**0.5)


def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol




#### Risk Kit ###

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4


import scipy.stats
def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal)
    else:
        statistic, p_value = scipy.stats.jarque_bera(r)
        return p_value > level


def drawdown(return_series: pd.Series):
    """Takes a time series of asset returns.
       returns a DataFrame with columns for
       the wealth index, 
       the previous peaks, and 
       the percentage drawdown
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})
def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame, else raises a TypeError
    """
    if isinstance(r, pd.Series):
        is_negative = r < 0
        return r[is_negative].std(ddof=0)
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(semideviation)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


from scipy.stats import norm
def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gauusian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))


#### Z scores and Z Score strategies  

# Rolling Z Score
def z_score(series,window):
    
    """
    Computes rolling Z-Score based on a set window 
    Parameters
    ----------
    series: Pandas Series
    
    window : int
    window to compute the  rolling Z-Score
    """
    r = series.rolling(window=window)
    m=r.mean().shift(1)
    s=r.std(ddof=0).shift(1)
    zs = (series-m)/s
  
    return zs



# Rolling Mapped Z Score - Transformation
def z_score_t(series,window):
    
    """
    Computes rolling Z-Score based on a set window with values bounded to 
    the range [-3, 3]

    Parameters
    ----------
    series: Pandas Series
    
    window : int
    window to compute the  rolling Z-Score
    """
    r = series.rolling(window=window)
    m=r.mean().shift(1)
    s=r.std(ddof=0).shift(1)
    zs = (series-m)/s

    zs [zs>3] = 3
    zs[zs <-3] =-3    
    return zs

def get_long_short(score):
    """
    Generate signals based on a score  long, short, and do nothing.
    
    Parameters
    ----------
    score : Score Series
        Score for each date
    
    Returns
    -------
    Signals : Series
        long(1), short (-1) do nothing (0)
    """
    
    #signals = pd.DataFrame(0, columns = close.columns, index = close.index) # Para Dataframe de varias series
    signals = pd.Series(0, index = score.index) # Para Serie 
    signals[1< score] = 1
    signals[-1 > score] = -1

    return signals


def clear_signals(signals, window_size):
    """
    Clear out signals in a series of just long or short signals.
    
    Reduce the number of signals down to 1 within the window size time period.
    
    Parameters
    ----------
    signals : Pandas Series 
        Long, Short or do nothing signals for each date
    window_size : int
        The number of days to have a single signal      
    
    Returns
    -------
    signals : Pandas Series
        Signals reduced to one per window
    """
    # Start with buffer of window size
    # This handles the edge case of calculating past_signal in the beginning
    clean_signals = [0]*window_size
    
    for signal_i, current_signal in enumerate(signals):
        # Check if there was a signal in the past window_size of days
        has_past_signal = bool(sum(clean_signals[signal_i:signal_i+window_size]))
        # Use the current signal if there's no past signal, else 0/False
        clean_signals.append(not has_past_signal and current_signal)
        
    # Remove buffer
    clean_signals = clean_signals[window_size:]

    # Return the signals as a Series of Ints
    return pd.Series(np.array(clean_signals).astype(np.int), signals.index)



def filter_signals(signal, lookahead_days):
    """
    Filter out signals in a Pandas Series
    
    Parametros
    ----------
    signal :Pandas Series
        Long, short, and do nothing signals for each date
    lookahead_days : int
        Días hacia adelante de cada ventana
    
    Returns
    -------
    filtered_signal : Pandas Serie
         Filtered long, short and do nothing signals for each date
    """
    #TODO: Implement function
    
    long_signal = pd.Series(0, index = signal.index)
    short_signal = pd.Series(0, index = signal.index)
    filter_l = pd.Series(0, index = signal.index)
    filter_s = pd.Series(0, index = signal.index)
    long_signal[signal>0] = 1
    short_signal[signal<0] = 1
      
   
      
    filter_l = clear_signals(long_signal, lookahead_days) 
    filter_s = -1*clear_signals(short_signal, lookahead_days)    
    
    
    return filter_l + filter_s



def get_lookahead_prices(close, lookahead_days):
    """
    Get the lookahead prices for `lookahead_days` number of days.
    
    Parameters
    ----------
    close : DataFrame
        Close price for each ticker and date
    lookahead_days : int
        The number of days to look ahead
    
    Returns
    -------
    lookahead_prices : DataFrame
        The lookahead prices for each ticker and date
    """
    #TODO: Implement function
    lookahead_prices = close.shift(-lookahead_days)
    
    return lookahead_prices

def get_diff_lookahead(close, lookahead_prices):
    """
    Calculate the log returns from the lookahead days to the signal day.
    
    Parameters
    ----------
    close : DataFrame
        Close price for each ticker and date
    lookahead_prices : DataFrame
        The lookahead prices for each ticker and date
    
    Returns
    -------
    lookahead_returns : DataFrame
        The lookahead log returns for each ticker and date
    """
    #TODO: Implement function
    lookahead_diff = lookahead_prices - close
    
    
    return lookahead_diff


def get_indicator(lookahead_diff, bps):
    """
    Generate indicator of  long movement, short movement, no movement.
    
    Parameters
    ----------
    lookahead_diff : Pandas Series
        Difference in price in signal window
    
    Returns
    -------
    indicator : Pandas Series
        1 for a difference larger than bps, -1 for a difference smaller than - bps
    """
    
    #signals = pd.DataFrame(0, columns = close.columns, index = close.index) # Para Dataframe de varias series
    indicator = pd.Series(0, index = lookahead_diff.index) # Para Serie 
    indicator[bps > lookahead_diff] = 1
    indicator[-bps <  lookahead_diff] = -1

    return indicator


def pred_success(target_series,series,zs_window,clean_window,lookahead_window,bps):
    
    """
    Count the number of times the prediction was correct in the window selected
    
    Parameters
    ----------
    Series : Pandas Series
        Series to compute z score and measure success
    zs_window: int     
        Number of days of the window for thr rooling ZS
    clean_window: ind
        Number of days of redundancy in a signal
    lookahead_window: int
        Number of days of the prediction
    bps: float
        Minimun movement size to be conisdered a long or short success
        
    
    Returns
    -------
    zs_window : int     
        Number of days of the window for thr rooling ZS
    succes: int
        Success indicator for the ZS window
    """
    
    
    
    zs = z_score_t(series,zs_window)
    signals =  get_long_short(zs)
    f_signals = filter_signals(signals,clean_window)
    lookahead_ind = get_lookahead_prices(target_series, lookahead_window)
    lookahead_ind_diff = get_diff_lookahead(target_series, lookahead_ind)
    indicator = get_indicator(lookahead_ind_diff, bps)
    success = f_signals * indicator
    
    
    
    return zs_window , success.sum()



def simulate_zs_windows(target_series,series,clean_window,lookahead_window,bps):
    
    """
    Count the number of times the prediction was correct in the window selected
    
    Parameters
    ----------
    Series : Pandas Series
        Series to compute z score and measure success
    clean_window: ind
        Number of days of redundancy in a signal
    lookahead_window: int
        Number of days of the prediction
    bps: float
        Minimun movement size to be conisdered a long or short success
        
    
    Returns
    -------
    simulations : DataFrame   
       Each ZS window and success indicator

    """
    

    zs_windows = list(range(250,1250,5))  # List of ZS windows to iterate 
    windows = []
    success= []
    simulations = pd.DataFrame()
    
    for window in zs_windows:
        simulation = pred_success(target_series,series,window,clean_window,lookahead_window,bps)
        windows.append( simulation[0])
        success.append( simulation[1])
        
    simulations['ZS_windows'] = windows  
    simulations['Success'] = success
    
    return simulations
    


####  Rates Markets functions

def hike(FF_df):
    
    """
    Computes te number of months until the nex rate hike implied by the FF curve
    

    Parameters
    ----------
    FF_df: Pandas DataFrame
    DataFrame with the Fed Funds Futures series
    
    Returns
    ---------
    hike : Pandas series
    The hike series
    """
    
    rate = 100 - FF_df
    rate_q = (rate*4).apply(np.ceil)/4 ## Get the correct hike band in 0.25 band increments
    
    hike = pd.Series()
    
    for index, row in rate_q.iterrows():
        counter = 0
        current_rate = row[0]
        
        for element in row :
            counter = counter +1
            if element > current_rate:
                break 
        hike[index] = counter
        
    return hike
