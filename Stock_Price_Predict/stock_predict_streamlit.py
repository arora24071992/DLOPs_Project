import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

# StockPredictor class and create_dataset function
class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, activation):
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.activation = activation

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.activation(x)
        return x

class StockPredictor(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super(StockPredictor, self).__init__()

        self.activation = activation()

        self.lstm1 = LSTMLayer(input_size=1, hidden_size=128, activation=self.activation)
        self.dropout1 = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.lstm2 = LSTMLayer(input_size=128, hidden_size=64, activation=self.activation)
        self.dropout2 = nn.Dropout(0.2)
        self.batch_norm2 = nn.BatchNorm1d(64)

        self.fc = nn.Linear(64, 1)
        self.dropout3 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.lstm1(x)
        x = self.dropout1(x)
        x = x.transpose(1, 2)
        x = self.batch_norm1(x)
        x = x.transpose(1, 2)
        x = self.lstm2(x)
        x = self.dropout2(x)
        x = x.transpose(1, 2)
        x = self.batch_norm2(x)
        x = x.transpose(1, 2)

        x = self.fc(x[:, -1, :])
        x = self.dropout3(x)
        return x



def create_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Title
st.title("Stock Price Prediction")

# Sidebar
st.sidebar.header("Select a stock ticker")
tickers = yf.Tickers(['AAPL','META', 'MSFT', 'AMZN', 'GOOGL', 'SBUX', 'TSLA'])
selected_ticker = st.sidebar.selectbox("Choose a ticker", tickers.tickers)


# Download historical stock prices
start_date = '2010-01-01'
end_date = '2022-01-01'
stock_data = yf.download(selected_ticker, start=start_date, end=end_date)

data = stock_data[['Close']]


# Normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
data['Close'] = scaler.fit_transform(data['Close'].values.reshape(-1, 1))


# Create a sliding window of the data
window_size = 100
X, y = create_dataset(data['Close'], window_size)

# Load the saved model
model = StockPredictor()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Create a navigation menu in the sidebar
menu = st.sidebar.selectbox("Navigation", ["Data", "Predictions"])

if menu == "Data":
    # Display stock data
    st.subheader(f"{selected_ticker} Stock Prices")
    st.write(stock_data)

elif menu == "Predictions":
    # Prepare the data
    X_test = torch.tensor(X[-1], dtype=torch.float32).unsqueeze(0).unsqueeze(2)

    # Plot the actual stock prices
    st.subheader(f"{selected_ticker} Stock Prices and Prediction")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(stock_data['Close'], label='Actual Prices')
    
    # Select the number of days to predict
    n_days = st.slider("Select number of days to predict:", 1,1, 10)
    if st.button("Predict"):
        # Make the predictions
        model.eval()
        with torch.no_grad():
            y_preds = []
            for _ in range(n_days):
                y_pred = model(X_test)
                y_pred = y_pred.unsqueeze(1).unsqueeze(2)
                y_pred = y_pred.reshape(X_test.shape[0], 1, 1, y_pred.shape[-1])
                y_preds.append(y_pred.item())
                y_pred_reshaped = y_pred.view(1, 1, 1)
                X_test = torch.cat((X_test[:, 1:, :], y_pred_reshaped), dim=1)

        # Inverse transform the predictions
        y_preds_transformed = scaler.inverse_transform(np.hstack([np.array(y_preds).reshape(-1, 1), np.zeros_like(np.array(y_preds).reshape(-1, 1))]))[:, 0]

        # Display the prediction
        st.write(f"Predicted prices for {selected_ticker} ({n_days} days ahead):")
        for i in range(n_days):
            st.write(f"Day {i+1}: ${y_preds_transformed[i]:.2f}")

        # Add the predicted prices to the plot
        predicted_dates = pd.date_range(stock_data.index[-1] + pd.DateOffset(days=1), periods=n_days)
        ax.plot(predicted_dates, y_preds_transformed, 'r-', label='Predicted Prices')

    ax.set_title(f'{selected_ticker} Stock Prices and Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

