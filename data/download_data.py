import yfinance as yf
import pandas as pd

ticker = "AAPL"

data = yf.download(ticker, start="2018-01-01", end="2024-01-01")

# reset index so Date becomes a column
data.reset_index(inplace=True)

data.to_csv("data/stock_data.csv", index=False)

print(data.head())