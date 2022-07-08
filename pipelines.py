import os
import pandas as pd
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression

import chart_studio.plotly as py
from chart_studio.plotly import plot, iplot
import plotly.graph_objs as go
import plotly.tools as plotly_tools
from plotly.subplots import make_subplots
import plotly.express as px

def current_date_stats_table(df, date_today):
    df = df[df['timeStamp']==date_today]
    desc_stats = descriptive_stats(df)
    fig = go.Figure(data=go.Table(
        columnwidth=[80, 80, 80, 80, 80],
        header=dict(values=['Date', 'Total Lessons', 'Max Speed', 'Average Speed', 'STD'],
        line_color='#d8a0a6',
        fill_color='#2b2b2c',
        align='center',
        font=dict(color='#d8a0a6'),
        height=40),
        cells=dict(values=[td, desc_stats['lesson_cnt'], desc_stats['top_spd']['spd'][0], desc_stats['avg_spd'], desc_stats['spd_std']],
        line_color='#d8a0a6',
        fill_color='#2b2b2c',        
        align='center',
        font=dict(color='#d8a0a6'),
        height=40),
    ))
    fig.update_layout(
        legend_title_side='top',
        showlegend=False,
        height=320,
        width=1080,
        plot_bgcolor='#2b2b2c',
    )
    fig.show()

def current_date_performance_graph(df:pd.DataFrame, date_today):
    
    df = df.loc[df['timeStamp'] == date_today]
    df = df.reset_index()

  
    
    # Descriptive Stats
    # total_time = sum(df['time'])/1000
    # lesson_cnt = len(df)
    # top_spd = max(df['speed'])/5
    # avg_spd = np.mean(df['speed'])/5
    # spd_std = np.std(df['speed']/5)

    # Scatter Plot of Speed (WPM) and Accuracy Rate
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    df['MA5'] = df.speed.rolling(5).mean()/5
    spd = df['speed']
    err = df['errors']/df['length']*100 #1 - df['errors']/df['length']

    regr = LinearRegression()
    regr.fit(np.array(df.index).reshape(-1,1), np.array(spd))
    fit = regr.predict(np.array(spd).reshape(-1,1))
    coef = regr.coef_
    if coef >= 0:
        col = '#00ff00'
    else:
        col = '#ff0000'

    fig.add_trace(go.Scatter(x=df.index + 1, y=spd, line=dict(width=2, color='#d8a0a6'), name='Test'))
    fig.add_trace(go.Scatter(x=df.index + 1, y=spd, mode='markers', marker=dict(size=8, color='#d8a0a6'), text=list(spd)))
    fig.add_trace(go.Scatter(x=df.index + 1, y=err, line=dict(width=2, color='#A9A9A9'), name='Test',), secondary_y=True)
    #fig.add_trace(go.Scatter(x=df.index + 1, y=err, mode='markers', marker=dict(size=8, color='#A9A9A9'), name='Test'), secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index + 1, y=fit, line=dict(width=2, dash='dot', color=col)))
    #fit.add_trace(go.Scatter())
    #fig.add_trace(go.Scatter(x=df.index + 1, y=accuracy, line=dict(width=2, color='red'), name='MV'))
    
    # Fig Layout
    fig.update_layout(
        legend_title_side='top',
        showlegend=False,
        height=800,
        width=1080,
        plot_bgcolor='#2b2b2c' 
    )
    fig.update_xaxes(title='Lessons', showgrid=False, range=[0.5,desc_stats['lesson_cnt']+0.5], dtick=5, tickangle=315, color='#76689a')
    fig.update_yaxes(title='Typing Speed (wpm)', range=[20,desc_stats['top_spd']['spd'][0]+10], color='#76689a', gridcolor='#76689a')
    fig.update_yaxes(title='Error Rate (%)', showgrid=False, secondary_y = True, range=[0,100], ticksuffix='%', color='#76689a')
    fig.show()
    #url = py.iplot(fig, filename='Typing Performance', auto_open=True)
    #return url
    fig.write_image("fig.svg")

def past_7d_trend(df, last_date):
    
    first_date = last_date - datetime.timedelta(days=6)
    df = df[df['timeStamp'] >= first_date]
    
    fig = px.box(df, x = 'timeStamp', y = 'speed', notched=True)
    fig.update_traces(marker = {
        'color':'#d8a0a6'
    })
    fig.update_xaxes(title='Date', showgrid=False, color='#76689a')
    fig.update_yaxes(title='Typing Speed (wpm)', range=[desc_stats['min_spd']['spd'][0]-8, desc_stats['top_spd']['spd'][0]+10], color='#76689a', gridcolor='#76689a')
    fig.update_layout(
        legend_title_side='top',
        showlegend=False,
        height=800,
        width=1080,
        plot_bgcolor='#2b2b2c',
    )
    fig.show()

def descriptive_stats(df:pd.DataFrame):
    desc_stats = {
        'total_time': datetime.timedelta(seconds=df.time.sum()/1000),
        'lesson_cnt': len(df),
        'top_spd': {
            'spd': df[df.speed == df.speed.max()].speed.values.round(2),
            'idx': df[df.speed == df.speed.max()].index.values
        },
        'min_spd': {
            'spd': df[df.speed == df.speed.min()].speed.values.round(2),
            'idx': df[df.speed == df.speed.min()].index.values
        }, 
        'avg_spd': round(np.mean(df['speed']), 2),
        'spd_std': round(np.std(df['speed']), 2)
    }
    return desc_stats
    
    # total_time = df.time.sum()/1000
    # lesson_cnt = len(df)
    # max_spd = [df[df.speed == df.speed.max()]['speed']/5, df[df.speed == df.speed.max()].index]
    # min_spd = min(df['speed'])/5
    # avg_spd = np.mean(df['speed'])/5
    # spd_std = np.std(df['speed']/5) 


def get_typing_data(raw_data_path:str) -> pd.DataFrame:
    df = pd.read_json(raw_data_path)
    return df


if __name__ == '__main__':
    RAW_DATA_PATH = './typing-data.json'
    # CERTIFICATE_PATH = 'Plotly Certificate/*.plotly.com.cer'
    # os.environ['REQUESTS_CA_BUNDLE'] = CERTIFICATE_PATH
    df = get_typing_data(RAW_DATA_PATH)
    td = datetime.datetime.today().date()
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    #current_day_report_generation(df, td)
    df['timeStamp'] = pd.to_datetime(df['timeStamp']).dt.date
    df['speed'] = df['speed'] / 5
    desc_stats = descriptive_stats(df)
    current_date_stats_table(df, yesterday)
    # current_date_performance_graph(df, td)
    # past_7d_trend(df, td)