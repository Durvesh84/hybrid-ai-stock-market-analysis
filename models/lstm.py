import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ===============================
# 1 Load dataset
# ===============================
df = pd.read_csv("data/stock_data.csv")

print("Original columns:", df.columns)

# ===============================
# 2 Remove text columns
# ===============================
df = df.drop(columns=["Date", "Ticker", "Price"], errors="ignore")

# ===============================
# 3 Convert everything to numeric
# ===============================
df = df.apply(pd.to_numeric, errors="coerce")

# ===============================
# 4 Remove NaN values
# ===============================
df = df.dropna()

print("Cleaned columns:", df.columns)

# ===============================
# 5 Extract Close price
# ===============================
data = df["Close"].values.reshape(-1,1)

# ===============================
# 6 Normalize data
# ===============================
scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)

# ===============================
# 7 Create sequences
# ===============================
X = []
y = []

sequence_length = 60

for i in range(sequence_length, len(data)):
    X.append(data[i-sequence_length:i])
    y.append(data[i])

X = np.array(X)
y = np.array(y)

print("Training shape:", X.shape)

# ===============================
# 8 Build LSTM model
# ===============================
model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

# ===============================
# 9 Train model
# ===============================
model.fit(X, y, epochs=10, batch_size=32)

print("LSTM training complete")