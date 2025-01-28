import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import yfinance as yf
import numpy as np
from datetime import date, timedelta
from keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split


def run_prediction():
    tickers = entry.get().split(',')
    today = date.today()
    d1 = today.strftime("%Y-%m-%d")
    end_date = d1
    d2 = date.today() - timedelta(days=1000)
    d2 = d2.strftime("%Y-%m-%d")
    start_date = d2
    predictions = {}

    for ticker in tickers:
        data = yf.download(ticker.strip(), start=start_date, end=end_date)
        if not data.empty:
            today_data = data.iloc[-1]

            features = np.array([[today_data['Open'], today_data['High'], today_data['Low'], today_data['Volume']]])
            features = features.reshape(features.shape[0], features.shape[1], 1)

            x = data[["Open", "High", "Low", "Volume"]]
            y = data["Close"]
            x = x.to_numpy()
            y = y.to_numpy()
            y = y.reshape(-1, 1)

            xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
            model = Sequential()
            model.add(LSTM(128, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
            model.add(LSTM(64, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            model.fit(xtrain, ytrain, batch_size=1, epochs=30, verbose=0)

            prediction = model.predict(features)
            predictions[ticker.strip()] = prediction[0][0]

    prediction_message = "\n".join([f"{ticker}: ${value:.2f}" for ticker, value in predictions.items()])
    messagebox.showinfo("Price Predictions", prediction_message)


root = tk.Tk()
root.title("Stock Price Prediction")

label = ttk.Label(root, text="Enter Stock Tickers (comma separated):")
label.pack(pady=10)

entry = ttk.Entry(root, width=50)
entry.pack(pady=10)

button = ttk.Button(root, text="Run Prediction", command=run_prediction)
button.pack(pady=20)

root.mainloop()

