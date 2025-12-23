import matplotlib.pyplot as plt
import os


import data_handler
import ml_engine

def plot_dashboard(historical_df, forecast_df):
    """Visualizes the data."""
    print("[Dashboard] Generating plot...")
    plt.figure(figsize=(12, 6))
    plt.plot(historical_df['Date'], historical_df['Sales'], label='Actual Sales', color='#1f77b4', alpha=0.7)
    plt.plot(forecast_df['Date'], forecast_df['Predicted_Sales'], label='AI Forecast', color='#ff7f0e', linestyle='--', linewidth=2)
    
    plt.title('AI-Powered Sales Forecasting Dashboard', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def save_deliverable(forecast_df):

    filename = "final_forecast_report.csv"
    forecast_df[['Date', 'Predicted_Sales']].to_csv(filename, index=False)
    print(f"[System] Report saved: {os.getcwd()}\\{filename}")

if __name__ == "__main__":

    data = data_handler.generate_mock_data()
    data_clean = data_handler.preprocess_data(data)
    
    model = ml_engine.train_model(data_clean)
    
    last_date = data_clean['Date'].max()
    forecast = ml_engine.make_forecast(model, last_date)
    
    save_deliverable(forecast)
    plot_dashboard(data_clean, forecast)