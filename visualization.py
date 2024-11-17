import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm

def plot_historical_vs_predicted(df):
    fig = make_subplots(rows=1, cols=1)
    
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Sales'],
                  mode='lines',
                  name='Actual Sales',
                  line=dict(color='#1f77b4'))
    )
    
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Predicted_Sales'],
                  mode='lines',
                  name='Predicted Sales',
                  line=dict(color='#ff7f0e'))
    )
    
    fig.update_layout(
        height=500,
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def plot_seasonal_pattern(df):
    monthly_avg = df.groupby('Month')['Sales'].mean().reset_index()
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=monthly_avg['Month'], 
                  y=monthly_avg['Sales'],
                  mode='lines+markers',
                  line=dict(color='#2ecc71'))
    )
    
    fig.update_layout(
        height=300,
        template='plotly_white',
        xaxis_title="Month",
        yaxis_title="Average Sales ($)",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def plot_components(df):
    decomposition = sm.tsa.seasonal_decompose(df['Sales'], period=30)
    
    fig_trend = go.Figure()
    fig_trend.add_trace(
        go.Scatter(x=df['Date'], y=decomposition.trend,
                  mode='lines',
                  name='Trend',
                  line=dict(color='#3498db'))
    )
    fig_trend.update_layout(
        height=300,
        template='plotly_white',
        title="Sales Trend",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    fig_seasonal = go.Figure()
    fig_seasonal.add_trace(
        go.Scatter(x=df['Date'], y=decomposition.seasonal,
                  mode='lines',
                  name='Seasonality',
                  line=dict(color='#2ecc71'))
    )
    fig_seasonal.update_layout(
        height=300,
        template='plotly_white',
        title="Seasonal Pattern",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig_trend, fig_seasonal
