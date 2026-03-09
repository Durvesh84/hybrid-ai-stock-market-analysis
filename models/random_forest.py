import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ======================
# 1 Load Data
# ======================
df = pd.read_csv("data/stock_data.csv")

print("Original Columns:", df.columns)

# ======================
# 2 Drop unwanted columns
# ======================
df = df.drop(columns=["Date", "Ticker", "Price"], errors="ignore")

# ======================
# 3 Convert everything to numeric
# This removes text like 'AAPL'
# ======================
df = df.apply(pd.to_numeric, errors='coerce')

# ======================
# 4 Remove rows with NaN
# ======================
df = df.dropna()

print("Cleaned Columns:", df.columns)

# ======================
# 5 Define Features
# ======================
X = df[["Open", "High", "Low", "Volume"]]
y = df["Close"]

# ======================
# 6 Train Test Split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False
)

# ======================
# 7 Train Model
# ======================
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ======================
# 8 Prediction
# ======================
pred = model.predict(X_test)

# ======================
# 9 Evaluation
# ======================
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("\nModel Performance")
print("------------------")
print("MSE:", mse)
print("R2:", r2)

# ======================
# 10 Save Model
# ======================
joblib.dump(model, "models/random_forest_model.pkl")

print("\nModel saved successfully")