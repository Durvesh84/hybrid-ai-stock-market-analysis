import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

from models.predictor import (
    random_forest_prediction,
    arima_prediction,
    lstm_prediction,
    hybrid_prediction
)

st.set_page_config(page_title="AI Stock Prediction", layout="wide")

st.title("📈 AI Stock Prediction Dashboard")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Controls")

ticker = st.sidebar.selectbox(
    "Select Stock",
    ["AAPL","TSLA","MSFT","NVDA","AMZN","META","GOOGL"]
)

model_choice = st.sidebar.selectbox(
    "Select Prediction Model",
    ["Random Forest","ARIMA","LSTM","Hybrid"]
)

# -----------------------------
# Download Stock Data
# -----------------------------
data = yf.download(ticker, period="1y")

current_price = float(data["Close"].iloc[-1])

# Moving averages
data["MA50"] = data["Close"].rolling(50).mean()
data["MA200"] = data["Close"].rolling(200).mean()

# -----------------------------
# Layout: Two Columns
# -----------------------------
col1, col2 = st.columns([2,1])

# -----------------------------
# Column 1: Graph
# -----------------------------
with col1:

    st.subheader(f"{ticker} Stock Performance")

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(data.index, data["Close"], label="Close Price")
    ax.plot(data.index, data["MA50"], label="MA50")
    ax.plot(data.index, data["MA200"], label="MA200")

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    ax.legend()

    st.pyplot(fig)

# -----------------------------
# Column 2: Prediction Panel
# -----------------------------
with col2:

    st.subheader("Prediction Panel")

    st.metric("Current Price", f"${current_price:.2f}")

    if st.button("Run Prediction"):

        if model_choice == "Random Forest":
            predicted_price = random_forest_prediction(ticker)

        elif model_choice == "ARIMA":
            predicted_price = arima_prediction(ticker)

        elif model_choice == "LSTM":
            predicted_price = lstm_prediction(ticker)

        elif model_choice == "Hybrid":
            predicted_price = hybrid_prediction(ticker)

        predicted_price = float(predicted_price)

        change = (predicted_price - current_price) / current_price * 100

        if change > 1:
            signal = "BUY 🟢"
        elif change < -1:
            signal = "SELL 🔴"
        else:
            signal = "HOLD 🟡"

        st.metric("Predicted Price", f"${predicted_price:.2f}")
        st.metric("Expected Change", f"{change:.2f}%")

        st.subheader(signal)