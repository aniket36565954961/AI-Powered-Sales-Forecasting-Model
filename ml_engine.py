import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_model(df):
    print("[AI] Training the prediction model...")
    
    features = ['Year', 'Month', 'Day', 'DayOfWeek', 'Quarter']
    X = df[features]
    y = df['Sales']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"[AI] Model Accuracy Check - MAE: {mae:.2f}")
    
    return model

def make_forecast(model, last_date, days_to_predict=90):
    print(f"[AI] Forecasting next {days_to_predict} days...")
    
    future_dates = [last_date + datetime.timedelta(days=x) for x in range(1, days_to_predict + 1)]
    future_df = pd.DataFrame({'Date': future_dates})
    
    future_df['Year'] = future_df['Date'].dt.year
    future_df['Month'] = future_df['Date'].dt.month
    future_df['Day'] = future_df['Date'].dt.day
    future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
    future_df['Quarter'] = future_df['Date'].dt.quarter
    
    future_df['Predicted_Sales'] = model.predict(future_df[['Year', 'Month', 'Day', 'DayOfWeek', 'Quarter']])
    
    return future_df