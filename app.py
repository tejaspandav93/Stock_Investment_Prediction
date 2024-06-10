import streamlit as st
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
    st.success(f"Trained XGBoost model saved as {filename}")

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

# Main function for Streamlit app
def main():
    st.title("Stock Price Prediction App")

    # Page navigation
    page = st.sidebar.selectbox("Page", ["Actual Value", "Prediction", "Ticker Information"])

    # Actual Value page
    if page == "Actual Value":
        st.header("Actual Value")
        symbol = st.text_input("Enter the stock symbol (e.g., AAPL):").upper()

        # Filter selection
        filter_option = st.selectbox("Select time period for the data:", ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "5 Years", "Max"])
        
        # Set date range based on filter
        if filter_option == "1 Week":
            start_date = datetime.now() - timedelta(weeks=1)
        elif filter_option == "1 Month":
            start_date = datetime.now() - timedelta(weeks=4)
        elif filter_option == "3 Months":
            start_date = datetime.now() - timedelta(weeks=12)
        elif filter_option == "6 Months":
            start_date = datetime.now() - timedelta(weeks=24)
        elif filter_option == "1 Year":
            start_date = datetime.now() - timedelta(days=365)
        elif filter_option == "5 Years":
            start_date = datetime.now() - timedelta(days=5*365)
        else:  # Max
            start_date = datetime(1900, 1, 1)  # A very early date to get all available data

        end_date = datetime.now()

        if st.button("Fetch Data"):
            data = fetch_stock_data(symbol, start_date, end_date)
            if not data.empty:
                st.write(data.tail())
                plt.figure(figsize=(10, 6))
                plt.plot(data.index, data['Close'], label='Actual Closing Prices')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.title(f'Actual Closing Prices for {symbol}')
                plt.legend()
                plt.grid(True)
                st.pyplot(plt)
            else:
                st.error("No data found for the given symbol and date range.")

    # Prediction page
    elif page == "Prediction":
        st.header("Prediction")
        symbol = st.text_input("Enter the stock symbol (e.g., AAPL):").upper()
        start_date = datetime.now() - timedelta(days=180)
        end_date = datetime.now()
        data = fetch_stock_data(symbol, start_date, end_date)
        X, y = prepare_data(data)
        model = train_model(X, y)
        future_dates, future_prices = predict_future_prices(model, data)
        predicted_data = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_prices})
        st.write(predicted_data)
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['Close'], label='Actual Closing Prices (Last 6 Months)', color='blue')
        plt.plot(predicted_data['Date'], predicted_data['Predicted Close'], label='Predicted Closing Prices (Next 30 Days)', color='red', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'Actual vs Predicted Closing Prices for {symbol}')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

    # Ticker Information page
    elif page == "Ticker Information":
        st.header("Ticker Information")
        symbol = st.text_input("Enter the stock symbol (e.g., AAPL):").upper()
        if st.button("Fetch Info"):
            ticker = yf.Ticker(symbol)
            info = ticker.info
            st.subheader(f"Information about {symbol}")
            st.write(f"**Name:** {info.get('longName', 'N/A')}")
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
            st.write(f"**Country:** {info.get('country', 'N/A')}")
            st.write(f"**Full Time Employees:** {info.get('fullTimeEmployees', 'N/A')}")
            st.write(f"**Business Summary:** {info.get('longBusinessSummary', 'N/A')}")
            st.write(f"**Market Cap:** {info.get('marketCap', 'N/A')}")
            st.write(f"**Enterprise Value:** {info.get('enterpriseValue', 'N/A')}")
            st.write(f"**Trailing P/E:** {info.get('trailingPE', 'N/A')}")
            st.write(f"**Forward P/E:** {info.get('forwardPE', 'N/A')}")
            st.write(f"**Price to Book:** {info.get('priceToBook', 'N/A')}")
            st.write(f"**PEG Ratio:** {info.get('pegRatio', 'N/A')}")
            st.write(f"**Price to Sales:** {info.get('priceToSalesTrailing12Months', 'N/A')}")
            st.write(f"**50-Day Moving Average:** {info.get('fiftyDayAverage', 'N/A')}")
            st.write(f"**200-Day Moving Average:** {info.get('twoHundredDayAverage', 'N/A')}")
            st.write(f"**Website:** {info.get('website', 'N/A')}")
            st.write(f"**Address:** {info.get('address1', 'N/A')}, {info.get('city', 'N/A')}, {info.get('state', 'N/A')}, {info.get('zip', 'N/A')}")

if __name__ == "__main__":
    main()
