import yfinance as yf
import pandas as pd
from xgboost import XGBRegressor
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Fetch historical stock data
def fetch_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Feature engineering
def create_features(data):
    data['SMA'] = data['Close'].rolling(window=20).mean()  # Simple Moving Average
    data['RSI'] = calculate_RSI(data['Close'])  # Relative Strength Index
    return data

# Calculate Relative Strength Index (RSI)
def calculate_RSI(close_prices, window=14):
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Prepare data for modeling
def prepare_data(data):
    data = create_features(data).dropna()
    X = data[['SMA', 'RSI']]
    y = data['Close']
    return X, y

# Train XGBoost model
def train_model(X_train, y_train):
    model = XGBRegressor()
    model.fit(X_train, y_train)
    return model

# Save the trained model
def save_model(model, filename):
    model.save_model(filename)
    print(f"Trained XGBoost model saved as {filename}")

# Make predictions for the next 30 days
def predict_future_prices(model, data):
    future_dates = pd.date_range(data.index[-1] + timedelta(days=1), periods=30, freq='B')
    future_prices = []
    for date in future_dates:
        X_pred = prepare_data_for_prediction(data)
        future_price = predict_stock_price(model, X_pred)
        future_prices.append(future_price[0])
        data.loc[date] = future_price[0]  # Append predicted price to data for next prediction
    return future_dates, future_prices

# Make predictions
def predict_stock_price(model, X):
    prediction = model.predict(X)
    return prediction

# Prepare data for prediction
def prepare_data_for_prediction(data):
    data = create_features(data).dropna().tail(1)  # Use only the latest data
    X = data[['SMA', 'RSI']]
    return X

def main():
    symbol = input("Enter the stock symbol (e.g., AAPL): ").upper()

    # Fetch historical stock data for the last 6 months
    start_date = datetime.now() - timedelta(days=180)
    end_date = datetime.today()
    data = fetch_stock_data(symbol, start_date, end_date)

    # Prepare data for modeling
    X, y = prepare_data(data)

    # Train XGBoost model
    model = train_model(X, y)

    # Evaluate the model
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    print(f"Mean Absolute Error (MAE): {mae}")

    # Save the trained model
    model_filename = "xgboost_model.model"
    save_model(model, model_filename)

    # Predict closing prices for the next 30 days
    future_dates, future_prices = predict_future_prices(model, data)

    # Display the predicted prices along with their dates
    predicted_data = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_prices})
    print(predicted_data)

    # Plot actual closing prices for the historical data and predicted closing prices for the future
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label='Actual Closing Prices (Last 6 Months)', color='blue')
    plt.plot(predicted_data['Date'], predicted_data['Predicted Close'], label='Predicted Closing Prices (Next 30 Days)', color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'Actual vs Predicted Closing Prices for {symbol}')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
