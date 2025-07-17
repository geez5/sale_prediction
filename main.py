import streamlit as st
import pandas as pd
from datetime import timedelta
from data_generator import generate_sample_data
from model import SalesPredictor
from svisualization import (
    plot_historical_vs_predicted,
    plot_seasonal_pattern,
    plot_components
)

st.set_page_config(page_title="Sales Forecasting", layout="wide")

# Page styling
st.markdown("""
    <style>
        .main { padding: 2rem; }
        .stPlotlyChart {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Sales Forecasting Dashboard")

# Generate and prepare data
df = generate_sample_data()
predictor = SalesPredictor()
df = predictor.prepare_features(df)

# Train model and get metrics
metrics = predictor.train(df)
df['Predicted_Sales'] = predictor.predict(df[['DayOfWeek', 'Month', 'Year', 'DayOfYear']])

# Create layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Historical Sales and Predictions")
    fig_historical = plot_historical_vs_predicted(df)
    st.plotly_chart(fig_historical, use_container_width=True)

with col2:
    st.subheader("Model Performance")
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        st.metric("R² Score", f"{metrics['r2']:.3f}")
    with metrics_col2:
        st.metric("RMSE", f"${metrics['rmse']:.2f}")
    
    st.subheader("Seasonal Pattern")
    fig_seasonal = plot_seasonal_pattern(df)
    st.plotly_chart(fig_seasonal, use_container_width=True)

# Future Forecast
st.subheader("Future Sales Forecast")

future_days = 90
future_dates = pd.date_range(
    start=df['Date'].max() + timedelta(days=1),
    periods=future_days, 
    freq='D'
)

future_df = pd.DataFrame({'Date': future_dates})
future_df = predictor.prepare_features(future_df)
future_df['Predicted_Sales'] = predictor.predict(
    future_df[['DayOfWeek', 'Month', 'Year', 'DayOfYear']]
)

# Components Analysis
st.subheader("Sales Components Analysis")
fig_trend, fig_seasonal = plot_components(df)

components_col1, components_col2 = st.columns(2)
with components_col1:
    st.plotly_chart(fig_trend, use_container_width=True)
with components_col2:
    st.plotly_chart(fig_seasonal, use_container_width=True)
