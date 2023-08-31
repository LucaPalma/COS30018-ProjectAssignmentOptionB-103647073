# File: stock_prediction.py
# Authors: Cheong Koo and Bao Vo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 25/07/2023 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following:
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import yfinance as yf
import os
import os.path

from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer
from sklearn.model_selection import train_test_split

scalerData = {}
PRICE_VALUE_UPPER = "Close"
PRICE_VALUE_LOWER = "Open"


# ------------------------------------------------------------------------------
def load_process_data(TRAIN_START, TRAIN_END, date_split=True, split_ratio=0.3, store_data=True
                      , scale_min=0, scale_max=1, store_scaler=True):
    # TRAIN_START is the starting date for the dataset as a y/m/d string e.g. '2022-01-01'.
    # TRAIN_END is the ending date for the dataset as a y/m/d string e.g. '2023-01-01'.
    # date_split specifies whether the date should be split by date (true) or randomly (false)
    # split_ratio specifies the ratio of test/train data to be used ie 0.3 = 30% test data.
    # store_data specifies whether the generated data will be stored to file or not.
    # scale_min specifies the minimum value of your scaler.
    # scale_max specifies the maximum value of your scaler.
    # store_scaler allows you to store your scaler for future use.

    DATA_SOURCE = "yahoo"
    COMPANY = "TSLA"

    ## TO DO:
    # 1) Check if data has been saved before.
    # If so, load the saved data
    # If not, save the data into a directory
    # ------------------------------------------------------------------------------
    data = {}
    # checks if data has already been generated, if not create it.
    if os.path.isfile("datafile"):
        print("Did find data")
        data = pd.read_csv("datafile")
    else:
        print("Didnt find data")
        data = yf.download(COMPANY, start=TRAIN_START, end=TRAIN_END, progress=False)
        # takes the inputs from the parameters passed into function to create range of data.
        # Generates a pandas data structure that contains dates ascociated with the elements listed in feature columns.

    # For more details:
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html
    # ------------------------------------------------------------------------------
    # Prepare Data
    ## To do:
    # 1) Check if data has been prepared before.
    # If so, load the saved data
    # If not, save the data into a directory
    # 2) Use a different price value eg. mid-point of Open & Close
    # 3) Change the Prediction days
    # ------------------------------------------------------------------------------
    result = {}

    data.dropna(inplace=True)
    # Drop Empty Data

    scaler = MinMaxScaler(feature_range=(scale_min, scale_max))
    scaled_data = scaler.fit_transform(
        (data[PRICE_VALUE_UPPER].values.reshape(-1, 1) + data[PRICE_VALUE_UPPER].values.reshape(-1, 1)) / 2)
    # Function stores scaler data for later use if needed.
    scalerData = {
        "minScaler": [scale_min],
        "maxScaler": [scale_max]
    }
    # Flatten and normalise the data
    # First, we reshape a 1D array(n) to 2D array(n,1)
    # We have to do that because sklearn.preprocessing.fit_transform()
    # requires a 2D array
    # Here n == len(scaled_data)
    # Then, we scale the whole array to the range (0,1)
    # The parameter -1 allows (np.)reshape to figure out the array size n automatically
    # values.reshape(-1, 1)
    # https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape'
    # When reshaping an array, the new shape must contain the same number of elements
    # as the old shape, meaning the products of the two shapes' dimensions must be equal.
    # When using a -1, the dimension corresponding to the -1 will be the product of
    # the dimensions of the original array divided by the product of the dimensions
    # given to reshape so as to maintain the same number of elements.

    # Number of days to look back to base the prediction
    PREDICTION_DAYS = 60  # Original

    """
        # To store the training data
    x_train = []
    y_train = []

    scaled_data = scaled_data[:, 0]  # Turn the 2D array back to a 1D array
            # Prepare the data

            # Convert them into an array
    x_train, y_train = np.array(x_train), np.array(y_train)
            # Now, x_train is a 2D array(p,q) where p = len(scaled_data) - PREDICTION_DAYS
            # and q = PREDICTION_DAYS; while y_train is a 1D array(p)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            # We now reshape x_train into a 3D array(p, q, 1); Note that x_train
            # is an array of p inputs with each input being a 2D array
    """

    X, y = [], []
    for x in range(PREDICTION_DAYS, len(scaled_data)):
        X.append(scaled_data[x - PREDICTION_DAYS:x])
        y.append(scaled_data[x])

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    if date_split:
        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - split_ratio) * len(X))
        # this line takes in the split ratio and finds its opposite value for train samples, it then multiplies by the amount of prediction days.
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"] = X[train_samples:]
        result["y_test"] = y[train_samples:]
    else:
        # split the dataset randomly
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                                    test_size=split_ratio,
                                                                                                    shuffle=False)
        # Uses train_test_split import which uses given ratio parameter and two numpy arrays to split another array randomly between them.
    x_train = X
    y_train = y
    # assigning x and y values to x_train and y_train for use in the lower model fitting.

    if store_data:
        data.to_csv("datafile")
    # if parameter set to true, stores the scaled data as a csv file.

    # ------------------------------------------------------------------------------
    # Build the Model
    ## TO DO:
    # 1) Check if data has been built before.
    # If so, load the saved data
    # If not, save the data into a directory
    # 2) Change the model to increase accuracy?
    # ------------------------------------------------------------------------------
    model = Sequential()  # Basic neural network
    # See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
    # for some useful examples

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    # This is our first hidden layer which also spcifies an input layer.
    # That's why we specify the input shape for this layer;
    # i.e. the format of each training example
    # The above would be equivalent to the following two lines of code:
    # model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
    # model.add(LSTM(units=50, return_sequences=True))
    # For som eadvances explanation of return_sequences:
    # https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
    # https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
    # As explained there, for a stacked LSTM, you must set return_sequences=True
    # when stacking LSTM layers so that the next LSTM layer has a
    # three-dimensional sequence input.

    # Finally, units specifies the number of nodes in this layer.
    # This is one of the parameters you want to play with to see what number
    # of units will give you better prediction quality (for your problem)

    model.add(Dropout(0.2))
    # The Dropout layer randomly sets input units to 0 with a frequency of
    # rate (= 0.2 above) at each step during training time, which helps
    # prevent overfitting (one of the major problems of ML).

    model.add(LSTM(units=50, return_sequences=True))
    # More on Stacked LSTM:
    # https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))
    # Prediction of the next closing value of the stock price

    # We compile the model by specify the parameters for the model
    # See lecture Week 6 (COS30018)
    model.compile(optimizer='adam', loss='mean_squared_error')
    # The optimizer and loss are two important parameters when building an
    # ANN model. Choosing a different optimizer/loss can affect the prediction
    # quality significantly. You should try other settings to learn; e.g.

    # optimizer='rmsprop'/'sgd'/'adadelta'/...
    # loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...

    # Now we are going to train this model with our training data
    # (x_train, y_train)
    model.fit(x_train, y_train, epochs=25, batch_size=32)
    # Other parameters to consider: How many rounds(epochs) are we going to
    # train our model? Typically, the more the better, but be careful about
    # overfitting!
    # What about batch_size? Well, again, please refer to
    # Lecture Week 6 (COS30018): If you update your model for each and every
    # input sample, then there are potentially 2 issues: 1. If you training
    # data is very big (billions of input samples) then it will take VERY long;
    # 2. Each and every input can immediately makes changes to your model
    # (a souce of overfitting). Thus, we do this in batches: We'll look at
    # the aggreated errors/losses from a batch of, say, 32 input samples
    # and update our model based on this aggregated loss.

    # TO DO:
    # Save the model and reload it
    # Sometimes, it takes a lot of effort to train your model (again, look at
    # a training data with billions of input samples). Thus, after spending so
    # much computing power to train your model, you may want to save it so that
    # in the future, when you want to make the prediction, you only need to load
    # your pre-trained model and run it on the new input for which the prediction
    # need to be made.

    load_test_data(TRAIN_START, TRAIN_END, COMPANY, data, model, scaler, PREDICTION_DAYS)


# ------------------------------------------------------------------------------
# Test the model accuracy on existing data
# ------------------------------------------------------------------------------
# Load the test data
def load_test_data(TRAIN_START, TRAIN_END, COMPANY, data, model, scaler, PREDICTION_DAYS):
    PRICE_VALUE = "Close"
    test_data = yf.download(COMPANY, start=TRAIN_START, end=TRAIN_END, progress=False)

    # The above bug is the reason for the following line of code
    test_data = test_data[1:]

    actual_prices = test_data[PRICE_VALUE].values

    total_dataset = pd.concat((data[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
    # We need to do the above because to predict the closing price of the fisrt
    # PREDICTION_DAYS of the test period [TEST_START, TEST_END], we'll need the 
    # data from the training period

    model_inputs = model_inputs.reshape(-1, 1)
    # TO DO: Explain the above line

    model_inputs = scaler.transform(model_inputs)
    # We again normalize our closing price data to fit them into the range (0,1)
    # using the same scaler used above 
    # However, there may be a problem: scaler was computed on the basis of
    # the Max/Min of the stock price for the period [TRAIN_START, TRAIN_END],
    # but there may be a lower/higher price during the test period 
    # [TEST_START, TEST_END]. That can lead to out-of-bound values (negative and
    # greater than one)
    # We'll call this ISSUE #2

    # TO DO: Generally, there is a better way to process the data so that we 
    # can use part of it for training and the rest for testing. You need to 
    # implement such a way

    # ------------------------------------------------------------------------------
    # Make predictions on test data
    # ------------------------------------------------------------------------------
    x_test = []
    for x in range(PREDICTION_DAYS, len(model_inputs)):
        x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    # TO DO: Explain the above 5 lines

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    # Clearly, as we transform our data into the normalized range (0,1),
    # we now need to reverse this transformation 
    # ------------------------------------------------------------------------------
    # Plot the test predictions
    ## To do:
    # 1) Candle stick charts
    # 2) Chart showing High & Lows of the day
    # 3) Show chart of next few days (predicted)
    # ------------------------------------------------------------------------------

    plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
    plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
    plt.title(f"{COMPANY} Share Price")
    plt.xlabel("Time")
    plt.ylabel(f"{COMPANY} Share Price")
    plt.legend()
    plt.show()

    # ------------------------------------------------------------------------------
    # Predict next day
    # ------------------------------------------------------------------------------

    real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f"Prediction: {prediction}")

    # A few concluding remarks here:
    # 1. The predictor is quite bad, especially if you look at the next day 
    # prediction, it missed the actual price by about 10%-13%
    # Can you find the reason?
    # 2. The code base at
    # https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
    # gives a much better prediction. Even though on the surface, it didn't seem 
    # to be a big difference (both use Stacked LSTM)
    # Again, can you explain it?
    # A more advanced and quite different technique use CNN to analyse the images
    # of the stock price changes to detect some patterns with the trend of
    # the stock price:
    # https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
    # Can you combine these different techniques for a better prediction??


# begin loading and processing.
load_process_data('2022-01-01', '2023-01-01', True, 0.3, True, 0, 1, True)
