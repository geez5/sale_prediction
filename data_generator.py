import pandas as pd
import numpy as np
from datetime import datetime

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
