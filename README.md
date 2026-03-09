# AI Stock Price Prediction Dashboard

## Overview

This project is an **AI-powered stock analysis and prediction system** developed as part of an **MSc Data Science and Big Data Analytics final year project**. The application uses multiple machine learning and time-series models to analyze historical stock data and predict future price movements.

The system provides an interactive **Streamlit dashboard** where users can select a stock and choose different prediction models to generate a **BUY / SELL / HOLD signal** along with predicted price values.

The dashboard also visualizes **historical stock performance with technical indicators** and displays predictions in **Indian Rupees (₹)**.

---

## Features

* Interactive **Streamlit dashboard**
* Stock data retrieval using **Yahoo Finance**
* Visualization of historical stock performance
* **Moving average indicators (MA50 & MA200)**
* Multiple prediction models:

  * Random Forest
  * ARIMA
  * LSTM
  * Hybrid Model (combination of models)
* **BUY / SELL / HOLD signal generation**
* Currency conversion from **USD to INR**
* Model selection and stock selection directly from the dashboard

---

## Project Architecture

```
User Input (Streamlit Dashboard)
        │
        ▼
Stock Data Collection (yfinance API)
        │
        ▼
Feature Engineering
        │
        ▼
Prediction Models
 ├── Random Forest
 ├── ARIMA
 ├── LSTM
 └── Hybrid Model
        │
        ▼
Prediction Output
        │
        ▼
Dashboard Visualization
```

---

## Project Structure

```
stock_ai_project
│
├── models
│   ├── random_forest_model.pkl
│   ├── predictor.py
│   ├── random_forest.py
│   ├── arima_model.py
│   └── lstm_model.py
│
├── sentiment
│   └── news_sentiment.py
│
├── data
│   └── stock_data.csv
│
├── dashboard.py
└── README.md
```

---

## Technologies Used

### Programming

* Python

### Libraries

* Streamlit
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* Statsmodels
* Matplotlib
* yfinance
* NLTK

### Tools

* VS Code
* GitHub

---

## Machine Learning Models

### Random Forest

A supervised machine learning algorithm used for regression that predicts stock prices based on historical features such as:

* Open price
* High price
* Low price
* Volume

### ARIMA

A time-series forecasting model used for analyzing and predicting future stock prices based on historical closing prices.

### LSTM

A deep learning model capable of learning complex time-series patterns in stock price movements.

### Hybrid Model

A weighted combination of multiple models:

```
Final Prediction =
0.4 × LSTM
+ 0.3 × Random Forest
+ 0.3 × ARIMA
```

This improves prediction accuracy by combining different modeling techniques.

---

## Installation

### 1. Clone the Repository

```
git clone https://github.com/yourusername/stock-ai-project.git
```

### 2. Navigate to the Project Folder

```
cd stock-ai-project
```

### 3. Create a Virtual Environment

```
python -m venv venv
```

### 4. Activate the Environment

Windows

```
venv\Scripts\activate
```

Mac/Linux

```
source venv/bin/activate
```

### 5. Install Dependencies

```
pip install -r requirements.txt
```

---

## Running the Dashboard

Run the Streamlit dashboard using:

```
streamlit run dashboard.py
```

The application will open in your browser at:

```
http://localhost:8501
```

---

## Example Workflow

1. Select a **stock ticker** from the sidebar
2. Choose a **prediction model**
3. View **historical stock chart**
4. Click **Run Prediction**
5. See predicted price and **BUY / SELL signal**

---

## Example Output

```
Stock: AAPL
Current Price: ₹15,842.21
Predicted Price: ₹15,210.10
Expected Change: -3.98%
Signal: SELL
```

---

## Future Improvements

* Real-time stock price streaming
* Advanced technical indicators (RSI, MACD)
* Sentiment analysis from financial news
* Portfolio recommendation system
* Deployment to cloud platforms

---

## Author

Durvesh Chaudhari
MSc Data Science and Big Data Analytics

---

## License

This project is developed for **educational and research purposes**.
