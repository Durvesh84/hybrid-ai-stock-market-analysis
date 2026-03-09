import joblib
import numpy as np
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# load trained Random Forest model
rf_model = joblib.load("models/random_forest_model.pkl")


def get_stock_data(ticker):

    data = yf.download(ticker, period="90d")

    data = data.dropna()

    latest = data.iloc[-1]

    features = np.array([
        latest["Open"],
        latest["High"],
        latest["Low"],
        latest["Volume"]
    ]).reshape(1, -1)

    current_price = float(latest["Close"])

    return data, features, current_price


# -------------------------
# Random Forest Prediction
# -------------------------
def random_forest_prediction(ticker):

    data, features, current_price = get_stock_data(ticker)

    predicted_price = rf_model.predict(features)[0]

    return predicted_price


# -------------------------
# ARIMA Prediction
# -------------------------
def arima_prediction(ticker):

    data, features, current_price = get_stock_data(ticker)

    model = ARIMA(data["Close"], order=(5,1,0))

    model_fit = model.fit()

    forecast = model_fit.forecast(steps=1)

    predicted_price = float(forecast.iloc[0])

    return predicted_price


# -------------------------
# LSTM Prediction (simple)
# -------------------------
def lstm_prediction(ticker):

    data, features, current_price = get_stock_data(ticker)

    # simple approximation if trained model not saved
    predicted_price = current_price * np.random.uniform(0.98, 1.02)

    return float(predicted_price)


# -------------------------
# Hybrid Model
# -------------------------
def hybrid_prediction(ticker):

    rf = random_forest_prediction(ticker)
    arima = arima_prediction(ticker)
    lstm = lstm_prediction(ticker)

    predicted_price = (0.4 * rf) + (0.3 * arima) + (0.3 * lstm)

    return float(predicted_price)