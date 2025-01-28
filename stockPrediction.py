from datetime import date, timedelta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from keras import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, LSTM

stock_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN']

today = date.today()
d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=1000)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

num_stocks = len(stock_tickers)
cols = num_stocks
rows = 2

fig = make_subplots(rows=rows, cols=cols, subplot_titles=stock_tickers)

today_data = {}

for i, ticker in enumerate(stock_tickers):
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    data["Date"] = data.index
    data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]

    row = i // cols + 1
    col = i % cols + 1

    fig.add_trace(
        go.Candlestick(x=data["Date"],
                       open=data["Open"],
                       high=data["High"],
                       low=data["Low"],
                       close=data["Close"],
                       name=ticker),
        row=row, col=col
    )

    today_data[ticker] = data.iloc[-1]

fig.update_layout(title="Stock Price Analysis", xaxis_rangeslider_visible=False, height=600 * rows)
fig.show()

with open("correlation_and_predictions.txt", "w") as file:
    for ticker in stock_tickers:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        correlation = data.corr()
        correlation_close = correlation["Close"].sort_values(ascending=False)

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

        today_features = np.array([[today_data[ticker]["Open"],
                                    today_data[ticker]["High"],
                                    today_data[ticker]["Low"],
                                    today_data[ticker]["Volume"]]])

        today_features = today_features.reshape(today_features.shape[0], today_features.shape[1], 1)

        prediction = model.predict(today_features)

        file.write(f"Stock: {ticker}\n")
        file.write("Correlation with Close Price:\n")
        file.write(correlation_close.to_string())
        file.write("\nCurrent-Day Close Prediction: ")
        file.write(str(prediction[0][0]))
        file.write("\n\n")
