import pandas as pd
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression

from chart_studio.plotly import plot, iplot
import plotly.graph_objs as go
import plotly.tools as plotly_tools
from plotly.subplots import make_subplots



def current_day_report_generation(df:pd.DataFrame, date_today):
    df['timeStamp'] = pd.to_datetime(df['timeStamp']).dt.date
    df = df.loc[df['timeStamp'] == date_today]
    df = df.reset_index()
    
    # Description Stats
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
    res = regr.fit(np.array(df.index).reshape(-1,1), np.array(spd))
    fit = regr.predict(np.array(spd).reshape(-1,1))
    coef = regr.coef_
    if coef >= 0:
        col = '#8FBC8F'
    else:
        col = '#FF7F50'

    fig.add_trace(go.Scatter(x=df.index + 1, y=spd, line=dict(width=2, color='#483D8B'), name='Test'))
    fig.add_trace(go.Scatter(x=df.index + 1, y=spd, mode='markers', marker=dict(size=8, color='#483D8B'), name='Test'))
    fig.add_trace(go.Scatter(x=df.index + 1, y=err, line=dict(width=2, color='#A9A9A9'), name='Test',), secondary_y=True)
    #fig.add_trace(go.Scatter(x=df.index + 1, y=err, mode='markers', marker=dict(size=8, color='#A9A9A9'), name='Test'), secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index + 1, y=fit, line=dict(width=2, dash='dot', color=col)))

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
    fig.update_yaxes(title='Typing Speed (wpm)', range=[20, top_spd+10])
    fig.update_yaxes(title='Error Rate (%)', showgrid=False, secondary_y = True, range=[0,100])
    fig.show()
    url = iplot(fig, filename='performance')


def get_typing_data(raw_data_path:str) -> pd.DataFrame:
    df = pd.read_json(raw_data_path)
    return df


if __name__ == '__main__':
    RAW_DATA_PATH = './typing-data.json'
    df = get_typing_data(RAW_DATA_PATH)
    td = datetime.datetime.today().date()
    report = current_day_report_generation(df, td)
