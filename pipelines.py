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



def current_day_report_generation(df:pd.DataFrame, date_today):
    df['timeStamp'] = pd.to_datetime(df['timeStamp']).dt.date
    df = df.loc[df['timeStamp'] == date_today]
    df = df.reset_index()

    desc_stats = descriptive_stats(df)    
    
    # Descriptive Stats
    total_time = sum(df['time'])/1000
    lesson_cnt = len(df)
    top_spd = max(df['speed'])/5
    avg_spd = np.mean(df['speed'])/5
    spd_std = np.std(df['speed']/5)

    # Scatter Plot of Speed (WPM) and Accuracy Rate
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    df['MA5'] = df.speed.rolling(5).mean()/5
    spd = df['speed']/5
    err = df['errors']/df['length']*100 #1 - df['errors']/df['length']

    regr = LinearRegression()
    regr.fit(np.array(df.index).reshape(-1,1), np.array(spd))
    fit = regr.predict(np.array(spd).reshape(-1,1))
    coef = regr.coef_
    if coef >= 0:
        col = '#8FBC8F'
    else:
        col = '#FF7F50'

    fig.add_trace(go.Scatter(x=df.index + 1, y=spd, line=dict(width=2, color='#483D8B'), name='Test'))
    fig.add_trace(go.Scatter(x=df.index + 1, y=spd, mode='markers', marker=dict(size=8, color='#483D8B'), text=list(spd)))
    fig.add_trace(go.Scatter(x=df.index + 1, y=err, line=dict(width=2, color='#A9A9A9'), name='Test',), secondary_y=True)
    #fig.add_trace(go.Scatter(x=df.index + 1, y=err, mode='markers', marker=dict(size=8, color='#A9A9A9'), name='Test'), secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index + 1, y=fit, line=dict(width=2, dash='dot', color=col)))
    #fit.add_trace(go.Scatter())
    #fig.add_trace(go.Scatter(x=df.index + 1, y=accuracy, line=dict(width=2, color='red'), name='MV'))
    
    # Fig Layout
    fig.update_layout(
        legend_title_side='top',
        showlegend=False,
        height=480,
        width=1080,
        plot_bgcolor='#FFFACD' 
    )
    fig.update_xaxes(title='Lessons', showgrid=False, range=[0.5,lesson_cnt+0.5], dtick=1, tickangle=315)
    fig.update_yaxes(title='Typing Speed (wpm)', range=[20,desc_stats['top_spd']['spd'][0]+10])
    fig.update_yaxes(title='Error Rate (%)', showgrid=False, secondary_y = True, range=[0,100], ticksuffix='%')
    fig.show()
    #py.plot(fig, filename='Typing Performance', auto_open=True)

def descriptive_stats(df:pd.DataFrame):
    desc_stats = {
        'total_time': datetime.timedelta(seconds=df.time.sum()/1000),
        'lesson_cnt': len(df),
        'top_spd': {
            'spd': df[df.speed == df.speed.max()].speed.values / 5,
            'idx': df[df.speed == df.speed.max()].index.values
        },
        'min_spd': {
            'spd': df[df.speed == df.speed.min()].speed.values / 5,
            'idx': df[df.speed == df.speed.min()].index.values
        }, 
        'avg_spd': np.mean(df['speed'])/5,
        'spd_std': np.std(df['speed'])/5
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
    CERTIFICATE_PATH = r'./Plotly Certificate/plotly_certificate.cer'
    os.environ['REQUESTS_CA_BUNDLE'] = CERTIFICATE_PATH
    df = get_typing_data(RAW_DATA_PATH)
    #td = datetime.datetime.today().date()
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    report = current_day_report_generation(df, yesterday)
