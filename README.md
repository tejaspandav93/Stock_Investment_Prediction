


## Stock Price Prediction Web Application

## Overview:
The Stock Price Prediction Web Application is a tool developed using Streamlit and Python to predict future stock prices based on historical data. It utilizes machine learning techniques, particularly the XGBoost regressor, to forecast stock prices and provides users with insights into potential investment opportunities. This README provides a detailed guide on setting up and using the application.

## Features
- **Fetch Historical Stock Data**: Retrieve historical stock data from Yahoo Finance API.
- **Data Preprocessing**: Perform feature engineering to create relevant features for model training.
- **Model Training**: Train an XGBoost regression model using historical stock data.
- **Prediction Visualization**: Visualize actual and predicted stock prices using interactive charts.
- **Ticker Information**: Fetch detailed information about a specific stock ticker, including sector, industry, market cap, etc.

## Requirements
- Python 3.x
- Streamlit
- yfinance
- pandas
- xgboost
- matplotlib

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/username/stock-price-prediction-web-app.git
   ```
2. Navigate to the project directory:
   ```bash
   cd stock-price-prediction-web-app
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit web application:
   ```bash
   streamlit run app.py
   ```
2. Access the application in your web browser at `http://localhost:8501`.

## Guide
### Actual Value Page
- Enter the stock symbol (e.g., AAPL).
- Select the time period for the data (e.g., 1 Week, 1 Month).
- Click the "Fetch Data" button to retrieve and visualize historical stock prices.

### Prediction Page
- Enter the stock symbol (e.g., AAPL).
- The application fetches historical stock data for the past 180 days by default.
- Train an XGBoost model on the historical data and predict future stock prices for the next 30 days.
- Visualize actual and predicted closing prices on an interactive chart.

### Ticker Information Page
- Enter the stock symbol (e.g., AAPL).
- Click the "Fetch Info" button to retrieve detailed information about the stock ticker, including sector, industry, market cap, etc.

## Directory Structure

stock-price-prediction-web-app/
│
├── app.py               # Main Streamlit application script
├── requirements.txt     # Python dependencies
├── README.md            # Project README file
└── 


## Contributing
Contributions are welcome! Please follow the [contribution guidelines](CONTRIBUTING.md).

## License
This project is licensed under the [MIT License](LICENSE).


Feel free to customize this README to fit your specific project requirements and provide more detailed instructions if necessary.
