# Stock Price Prediction

This project uses a Long Short-Term Memory (LSTM) neural network to predict stock prices. The model is built using PyTorch and the app interface is created using Streamlit. Stock prices are fetched using the yfinance library.

## Dependencies

To install the required dependencies, run:
pip install -r requirements.txt

## Usage

To run the app, execute the following command:
streamlit run app.py


This will open the app in your default web browser.

In the app, you can:

1. Select a stock ticker from the sidebar.
2. View the historical stock prices by choosing "Data" in the navigation menu.
3. Predict the stock prices for the selected number of days ahead by choosing "Predictions" in the navigation menu and adjusting the slider.

The model predicts the stock prices and displays the actual and predicted prices on a plot.

## Files

- `app.py`: The main app file containing the Streamlit app and the LSTM model.
- `model.pth`: The saved model file containing the trained weights of the LSTM model.
- `requirements.txt`: The list of required dependencies to run the app.

## Note

This project is for educational purposes only and should not be used for financial decisions or trading.

Diagram:

<img width="1792" alt="image" src="https://user-images.githubusercontent.com/111610085/236722217-a4b4a9d6-172d-4446-a8c7-613a9af8cb23.png">

