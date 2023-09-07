# File: stock_prediction.pyV.03
# Author: Luca Palma 103647073

# Original Code
# Authors: Cheong Koo and Bao Vo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 25/07/2023 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# YouTube link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import yfinance as yf
import os
import os.path
import pickle

from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer
from sklearn.model_selection import train_test_split

# Parameters

# General
COMPANY = "TSLA"
PREDICTION_DAYS = 50
PRICE_VALUE_UPPER = "Close"
PRICE_VALUE_LOWER = "Open"

# Display Parameters
displayType = "boxplot"  # Can be set to graph, candle or boxplot
displayTradingDays = 25  # from task B.3 select n trading days for chart to display.

# Data loading and processing Parameters
startdate = '2022-01-01'
enddate = '2023-01-01'
splitByDate = True
splitRatio = 0.3
dataStore = True
scaleMin = 0
scaleMax = 1

# Model building parameters
Cell_Units = 50
Dense_Units = 1
Dropout_Amt = 0.2
Epoch_Count = 25
Batch_Count = 32
Layer_Name = "DeepLearningLayer"
modelType = LSTM
Layer_Number = 2


def load_process_data(TRAIN_START, TRAIN_END, date_split=True, split_ratio=0.3, store_data=True,
                      scale_min=0, scale_max=1):
    if os.path.isfile("datafile"):
        print("Did find data")
        data = pd.read_csv("datafile")
    else:
        print("Didn't find data")
        data = yf.download(COMPANY, start=TRAIN_START, end=TRAIN_END, progress=False)

    result = {}

    data.dropna(inplace=True)
    # Drop Empty Data

    scaler = MinMaxScaler(feature_range=(scale_min, scale_max))
    scaled_data = scaler.fit_transform(
        (data[PRICE_VALUE_UPPER].values.reshape(-1, 1) + data[PRICE_VALUE_UPPER].values.reshape(-1, 1)) / 2)

    with open('minmax_scaler.pkl', 'wb') as f:
        pickle.dump(scaled_data, f)
    # Save the scaler to file.

    X, y = [], []
    for x in range(PREDICTION_DAYS, len(scaled_data)):
        X.append(scaled_data[x - PREDICTION_DAYS:x])
        y.append(scaled_data[x])

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    if date_split:
        train_samples = int((1 - split_ratio) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"] = X[train_samples:]
        result["y_test"] = y[train_samples:]
    else:
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(
            X, y, test_size=split_ratio, shuffle=False)
    x_train = X
    y_train = y

    if store_data:
        data.to_csv("datafile")

    # model = build_model(x_train, y_train, Cell_Units, Dense_Units, Dropout_Amt, Epoch_Count, Batch_Count, Layer_Name)

    model = build_customModel(x_train, y_train, Cell_Units, Dense_Units, Dropout_Amt,
                              Epoch_Count, Batch_Count, Layer_Name, modelType, Layer_Number)

    load_test_data(TRAIN_START, TRAIN_END, COMPANY, data, model, scaler, PREDICTION_DAYS,
                   displayType)


def build_model(x_train, y_train, unitsLSTM, unitsDense, dropOutAmt, epochAmt, batchAmt, layerName):
    model = Sequential()  # Basic neural network
    model.add(LSTM(units=unitsLSTM, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(dropOutAmt))

    model.add(LSTM(units=unitsLSTM, return_sequences=True))
    model.add(Dropout(dropOutAmt))

    model.add(LSTM(units=unitsLSTM))
    model.add(Dropout(dropOutAmt))

    model.add(Dense(units=unitsDense))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=epochAmt, batch_size=batchAmt)

    # Update layer names
    for layer in model.layers:
        layer._name = layer.name + layerName
    return model


def build_customModel(x_train, y_train, unitsCell, unitsDense, dropOutAmt, epochAmt,
                      batchAmt, layerName, cell, layerNumber):
    model = Sequential()
    for i in range(layerNumber):
        if i == 0:
            # Runs on the first layer of the model, uses trained data.
            model.add(cell(units=unitsCell, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        elif i == layerNumber - 1:
            # Runs on the last layer of the model.
            model.add(cell(units=unitsCell, return_sequences=False))
        else:
            # Hidden layers
            model.add(cell(units=unitsCell, return_sequences=True))

        # Add dropout following each layer
        model.add(Dropout(dropOutAmt))

    model.add(Dense(units=unitsDense))
    # Add a Dense layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=epochAmt, batch_size=batchAmt)
    return model


# ------------------------------------------------------------------------------
# Test the model accuracy on existing data
# ------------------------------------------------------------------------------
# Load the test data
def load_test_data(TRAIN_START, TRAIN_END, COMPANY, data, model, scaler, PREDICTION_DAYS, chart_type):
    PRICE_VALUE = "Close"
    test_data = yf.download(COMPANY, start=TRAIN_START, end=TRAIN_END, progress=False)
    test_data = test_data[1:]
    actual_prices = test_data[PRICE_VALUE].values
    total_dataset = pd.concat((data[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)
    x_test = []

    for x in range(PREDICTION_DAYS, len(model_inputs)):
        x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    if chart_type == "graph":
        displayChart(actual_prices, predicted_prices, COMPANY)
    elif chart_type == "candle":
        displayCandle(data, COMPANY)
    elif chart_type == "boxplot":
        displayBoxPlot(data, COMPANY)

    real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f"Prediction: {prediction}")


def displayChart(actual_prices, predicted_prices, COMPANY):
    plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
    plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
    plt.title(f"{COMPANY} Share Price")
    plt.xlabel("Time")
    plt.ylabel(f"{COMPANY} Share Price")
    plt.legend()
    plt.show()


def displayCandle(data, COMPANY):
    plt.style.use("fivethirtyeight")
    # applies more visually interesting theme.

    chart_small = data[-displayTradingDays:]  # set number of days to display

    green_df = chart_small[chart_small.Close > chart_small.Open].copy()
    # create new chart for green values where closing value is greater than open, ie net gain.
    green_df["Height"] = green_df["Close"] - green_df["Open"]
    # Get the height value based on difference of close and open value.

    red_df = chart_small[chart_small.Close < chart_small.Open].copy()
    red_df["Height"] = red_df["Open"] - red_df["Close"]
    # Same as previous lines except finding net losses.

    plt.figure(figsize=(15, 7))  # create new figure for chart and set its size
    # Grey Lines

    plt.vlines(x=chart_small["Date"], ymin=chart_small["Low"], ymax=chart_small["High"], color="grey")
    # Set min and max of chart using low and high values from data, set line color to grey.

    # Green Candles
    plt.bar(x=green_df["Date"], height=green_df["Height"], bottom=green_df["Open"], color="Green")
    # Draw green candles based on separated data, change drawing color to green.

    # Red Candles
    plt.bar(x=red_df["Date"], height=red_df["Height"], bottom=red_df["Close"], color="Red")
    # Draw green candles based on separated data, change drawing
    # color to red. Uses close instead of open as it is opposite direction on chart.

    plt.yticks(range(100, 240, 20), ["{} $".format(v) for v in range(100, 240, 20)])
    # formats y-axis to display $ and increment in ticks of 20 between 100-240
    plt.xticks(rotation=75)
    plt.subplots_adjust(bottom=0.25)  # Make sure dates don't get cutoff screen
    # rotate dates for easier reading.

    plt.xlabel("Date")
    plt.ylabel(f"{COMPANY} Share Price")
    plt.title("Candlestick Chart")
    # updating labels and title of chart

    plt.show()


def displayBoxPlot(data, COMPANY):
    chart_small = data[-displayTradingDays:]  # set number of days to display

    plt.figure(figsize=(10, 15))
    xvalues = ["Open", "Close", "Low", "High"]
    plot = plt.boxplot(data[xvalues], patch_artist=True)
    # Open, Close, Low, High, Volume

    plt.yticks(range(100, 500, 20), ["{} $".format(v) for v in range(100, 500, 20)])
    plt.xticks((1, 2, 3, 4), xvalues)

    plt.ylabel(f"{COMPANY} Share Price")

    plt.title(f"Boxplot Chart for last {displayTradingDays} days.")
    plt.subplots_adjust(bottom=0.25)  # Make sure dates don't get cutoff screen
    # update labels, layout and title

    colours = ['g', 'b', 'orange', 'r']
    for patch, color in zip(plot['boxes'], colours):
        patch.set_facecolor(color)

    plt.grid(color='grey')

    plt.show()


# Call first function
load_process_data(startdate, enddate, splitByDate, splitRatio, dataStore, scaleMin, scaleMax)
