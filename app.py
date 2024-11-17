import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import statsmodels.api as sm

st.set_page_config(page_title="Sales Forecasting", layout="wide")

# Page styling
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stPlotlyChart {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Sales Forecasting Dashboard")

# Generate sample data
def generate_sample_data(n_points=365):
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='D')
    
    # Base trend
    trend = np.linspace(1000, 2000, n_points)
    
    # Seasonal component (yearly)
    seasonal = 200 * np.sin(2 * np.pi * np.arange(n_points) / 365)
    
    # Weekly pattern
    weekly = 100 * np.sin(2 * np.pi * np.arange(n_points) / 7)
    
    # Random noise
    noise = np.random.normal(0, 50, n_points)
    
    # Combine components
    sales = trend + seasonal + weekly + noise
    
    return pd.DataFrame({
        'Date': dates,
        'Sales': sales
    })

# Load or generate data
df = generate_sample_data()

# Prepare features for ML model
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['DayOfYear'] = df['Date'].dt.dayofyear

# Train ML model
features = ['DayOfWeek', 'Month', 'Year', 'DayOfYear']
X = df[features]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
df['Predicted_Sales'] = model.predict(X)

# Calculate metrics
r2 = r2_score(y_test, model.predict(X_test))
rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

# Create visualizations
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Historical Sales and Predictions")
    
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
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Model Performance")
    
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        st.metric("R² Score", f"{r2:.3f}")
    
    with metrics_col2:
        st.metric("RMSE", f"${rmse:.2f}")
    
    # Seasonal Pattern
    st.subheader("Seasonal Pattern")
    monthly_avg = df.groupby('Month')['Sales'].mean().reset_index()
    
    fig_seasonal = go.Figure()
    fig_seasonal.add_trace(
        go.Scatter(x=monthly_avg['Month'], 
                  y=monthly_avg['Sales'],
                  mode='lines+markers',
                  line=dict(color='#2ecc71'))
    )
    
    fig_seasonal.update_layout(
        height=300,
        template='plotly_white',
        xaxis_title="Month",
        yaxis_title="Average Sales ($)",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(fig_seasonal, use_container_width=True)

# Future Forecast
st.subheader("Future Sales Forecast")

# Generate future dates
future_days = 90
future_dates = pd.date_range(start=df['Date'].max() + timedelta(days=1), 
                           periods=future_days, freq='D')

future_df = pd.DataFrame({
    'Date': future_dates,
    'DayOfWeek': future_dates.dayofweek,
    'Month': future_dates.month,
    'Year': future_dates.year,
    'DayOfYear': future_dates.dayofyear
})

# Make future predictions
future_df['Predicted_Sales'] = model.predict(future_df[features])

# Plot future forecast
fig_forecast = go.Figure()

# Historical data
fig_forecast.add_trace(
    go.Scatter(x=df['Date'], y=df['Sales'],
              mode='lines',
              name='Historical Sales',
              line=dict(color='#1f77b4'))
)

# Future forecast
fig_forecast.add_trace(
    go.Scatter(x=future_df['Date'], y=future_df['Predicted_Sales'],
              mode='lines',
              name='Forecast',
              line=dict(color='#e74c3c'))
)

fig_forecast.update_layout(
    height=400,
    hovermode='x unified',
    template='plotly_white',
    showlegend=True,
    xaxis_title="Date",
    yaxis_title="Sales ($)",
    margin=dict(l=0, r=0, t=30, b=0)
)

st.plotly_chart(fig_forecast, use_container_width=True)

# Confidence intervals using statsmodels
decomposition = sm.tsa.seasonal_decompose(df['Sales'], period=30)

st.subheader("Sales Components Analysis")

components_col1, components_col2 = st.columns(2)

with components_col1:
    # Trend component
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
    st.plotly_chart(fig_trend, use_container_width=True)

with components_col2:
    # Seasonal component
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
    st.plotly_chart(fig_seasonal, use_container_width=True)
