import pandas as pd
import numpy as np

def generate_mock_data(days=730):
    print("[Data] Generating synthetic records...")
    dates = pd.date_range(start="2023-01-01", periods=days)
    
    # Trends and Seasonality
    trend = np.linspace(100, 300, days)
    seasonality = np.sin(np.linspace(0, 3.14 * 20, days)) * 50
    noise = np.random.normal(0, 20, days)
    sales = trend + seasonality + noise + 200
    
    data = pd.DataFrame({'Date': dates, 'Sales': sales})
    data['Sales'] = data['Sales'].clip(lower=0)
    return data

def preprocess_data(df):
    print("[Data] Preprocessing and extracting features...")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter
    
    return df