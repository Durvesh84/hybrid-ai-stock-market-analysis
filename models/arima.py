import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# =========================
# 1 Load dataset
# =========================
df = pd.read_csv("data/stock_data.csv")

print("Original columns:", df.columns)

# =========================
# 2 Remove unwanted columns
# =========================
df = df.drop(columns=["Date", "Ticker", "Price"], errors="ignore")

# =========================
# 3 Convert data to numeric
# =========================
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

# =========================
# 4 Remove missing values
# =========================
df = df.dropna()

# =========================
# 5 Create time series
# =========================
close_prices = df["Close"]

print("Data type of Close:", close_prices.dtype)

# =========================
# 6 Train ARIMA model
# =========================
model = ARIMA(close_prices, order=(5,1,0))

model_fit = model.fit()

# =========================
# 7 Forecast next 5 values
# =========================
forecast = model_fit.forecast(steps=5)

print("\nNext 5 predicted prices:")
print(forecast)