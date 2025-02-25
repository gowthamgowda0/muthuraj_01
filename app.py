import streamlit as st
from datetime import date
import yfinance as yf
import math
from plotly import graph_objs as go
import numpy as np
from tensorflow.keras.layers import Dropout, LSTM, Dense
from tensorflow.keras.models import Sequential
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse


def plotRawData(data1):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data1['Date'], y=data1['Close'], name='Stock Close Price'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def plotPredictionData(data1):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data1['Date'], y=data1['Close'] / 3, name='Stock Close Price'))
    fig.layout.update(title_text="Prediction Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("AAPL", "MSFT", "INFY")
selectedStock = st.selectbox("Select dataset for prediction", stocks)

years = st.slider("Years of prediction:", 1, 4)
period = years * 365

@st.cache_data
def loadData(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data = loadData(selectedStock)
st.subheader("Raw Data")
st.write(data.tail())

plotRawData(data)

# Data Preprocessing
train_dates = pd.to_datetime(data['Date'])
cols = list(data)[1:6]
df_for_training = data[cols].astype(float)

scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

trainX, trainY = [], []
n_future, n_past = 1, 14

for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

# LSTM Model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True),
    LSTM(64, activation='relu', return_sequences=True),
    LSTM(32, activation='relu', return_sequences=False),
    Dropout(0.2),
    Dense(trainY.shape[1])
])

model.compile(optimizer='adam', loss='mse')
model.fit(trainX, trainY, epochs=10, batch_size=16, validation_split=0.1, verbose=0)

# Forecasting
n_future = period
forecast_period_dates = pd.date_range(list(train_dates)[0], periods=n_future, freq='1d').tolist()
forecast = model.predict(trainX[-n_future:])

forecast_copies = np.repeat(forecast, df_for_training.shape[1], axis=1)
y_pred_future = scaler.inverse_transform(forecast_copies)[:, 0]

forecast_dates = [time_i.date() for time_i in forecast_period_dates]

df_forecast = pd.DataFrame({'Date': np.array(forecast_dates), 'Close': y_pred_future})
df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])

plotPredictionData(df_forecast)

# Calculate Error
a = data['Close'][:period]
b = df_forecast['Close'] / 3
st.write(f"Mean Squared Error for {selectedStock}: {math.sqrt(mse(a, b)):.2f}")
