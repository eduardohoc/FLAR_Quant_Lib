import helper
import scipy.stats

import numpy as np
import pandas as pd
import plotly as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.offline as offline_py
import plotly.express as px


offline_py.init_notebook_mode(connected=True)


### Trading strategies visualization functions

def _generate_security_trace(prices):
    return go.Scatter(
        name='Index',
        x=prices.index,
        y=prices,
        line={'color': helper.color_scheme['main_line']})



def plot_security(prices, title):
    config = helper.generate_config()
    layout = go.Layout(title=title)

    stock_trace = _generate_security_trace(prices)

    offline_py.iplot({'data': [stock_trace], 'layout': layout}, config=config)
    
    

def _generate_buy_annotations(prices, signal):
    return [{
        'x': index, 'y': price, 'text': 'Long', 'bgcolor': helper.color_scheme['background_label'],
        'ayref': 'y', 'ax': 0, 'ay': 5}
        for index, price in prices[signal == 1].iteritems()]


def _generate_sell_annotations(prices, signal):
    return [{
        'x': index, 'y': price, 'text': 'Short', 'bgcolor': helper.color_scheme['background_label'],
        'ayref': 'y', 'ax': 0, 'ay': 5}
        for index, price in prices[signal == -1].iteritems()]

   

def plot_security(prices, title):
    config = helper.generate_config()
    layout = go.Layout(title=title)

    stock_trace = _generate_security_trace(prices)

    offline_py.iplot({'data': [stock_trace], 'layout': layout}, config=config)


def plot_high_low(prices, lookback_high, lookback_low, title):
    config = helper.generate_config()
    layout = go.Layout(title=title)

    stock_trace = _generate_security_trace(prices)
    high_trace = go.Scatter(
        x=lookback_high.index,
        y=lookback_high,
        name='lookback_high',
        line={'color': helper.color_scheme['major_line']})
    low_trace = go.Scatter(
        x=lookback_low.index,
        y=lookback_low,
        name='lookback_low',
        line={'color': helper.color_scheme['minor_line']})

    offline_py.iplot({'data': [stock_trace, high_trace, low_trace], 'layout': layout}, config=config)



def plot_signal(price, signal, title):
    config = helper.generate_config()
    buy_annotations = _generate_buy_annotations(price, signal)
    sell_annotations = _generate_sell_annotations(price, signal)
    layout = go.Layout(
        title=title,
        annotations=buy_annotations + sell_annotations)

    stock_trace = _generate_security_trace(price)

    offline_py.iplot({'data': [stock_trace], 'layout': layout}, config=config)




### Market analysis and scorecard visualization functions
    
    


# Plot two securities  

def plot_two_tickers(prices_1, prices_2, y1_label, y2_label, x_axis_label, title):
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    
    fig.add_trace(
        go.Scatter(x=prices_1.index, 
                   y=prices_1, 
                   name= y1_label),
        secondary_y=False)
    
    fig.add_trace(
        go.Scatter(x=prices_2.index,
                   y=prices_2,
                   name= y2_label),
        secondary_y=True)
    
    # Add figure title
    fig.update_layout(
        title_text=title
    )
    
    # Set x-axis title
    fig.update_xaxes(title_text=  x_axis_label)
    
    # Set y-axes titles
    fig.update_yaxes(title_text= y1_label, secondary_y=False)
    fig.update_yaxes(title_text= y2_label, secondary_y=True)
    fig.show()




## Plot Score     
    
def plot_score(prices_1, score, title1, title2):
    
    fig=make_subplots(
        specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x= score.index,
            y= score,
            name="Score",
            marker_color='orange'), 
        secondary_y=False)
        
    fig.add_trace(
        go.Scatter(
        x=prices_1.index,
        y=prices_1,
        name = title1,
        line_color='darkblue'), 
        secondary_y=True)
    
    
    
    #fig.update_traces(texttemplate='%{text:.2s}')
    fig.update_layout(title_text = title2)
    
    # Set y-axes titles
    fig.update_yaxes(title_text= "Score", secondary_y=False)
    fig.update_yaxes(title_text= title1, secondary_y=True)
    
    fig.show()
    
    



# Plot Score Heathmap con anotaciones 
def plot_score_heatmap(score_data, title_1):

    dates = score_data.index.strftime("%m/%d/%y")
    score_categories = score_data.columns.values
    
    
    fig = px.imshow(score_data.round(decimals = 1),
                    text_auto=True, 
                    x=score_categories,
                    y=dates,
                    origin = 'lower',
                    color_continuous_scale='RdYlGn',
                    width=1000, 
                    height=500,
                    aspect="auto")
    
    # Add figure title
    fig.update_layout(title= title_1)
    
    
    fig.show()
   

    
def plot_histogram(df,Title):
    
    fig = px.histogram(df, x=Title, nbins=100)
    
    fig.show()