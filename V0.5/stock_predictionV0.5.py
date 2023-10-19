# File: stock_prediction.pyV.04
# Author: Luca Palma 103647073

# Original Code
# Authors: Cheong Koo and Bao Vo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 25/07/2023 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# YouTube link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine



import math
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
import seaborn as sns

from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer, SimpleRNN, GRU
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from pickle import dump, load
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# Parameters
# General
company = "TSLA"
predictionDays = 25

# Display Parameters
displayType = ""  # Can be set to graph, candle or boxplot
displayTradingDays = 100  # from task B.3 select n trading days for chart to display.

# Data loading and processing Parameters    
FEATURES = ['High','Low','Open','Close','Volume']#Selected features of data to use for training.
startDate = '2021-05-01'
endDate = '2021-10-01'
predictDate = '2021-10-10' #Should be same amount of days in future as outputCount
splitByDate = True
splitRatio = 0.8
dataStore = True
scaleMin = 0
scaleMax = 1

# Model building parameters
modelType = GRU # Name of layer can be LSTM, SimpleRNN, GRU etc
modelString = "GRU"
cellUnits = 300 # Number of units(neurons) in the chosen layer.
denseUnits = 8  # Number of units(neurons) in dense layers. Should be equal to number of features
dropoutAmt = 0.1 # Frequency of input units being set to 0 to reduce overfitting.
epochCount = 32 # Number of repetitions or epochs
batchCount = 8  # Amount of input samples trained at a time
layerNumber = 3  # Number of layers in the model.
outputCount = 7 # Number of days to predict ahead.

#Sarima model parameters.
seasonalRegression = 1
seasonalIntegrated = 2
seasonalMovingAverage = 3
season = 9
weight = 0.7

# Global Storage
normal = MinMaxScaler()
pred = MinMaxScaler()
global rawData
rawData = pd.DataFrame()
global data_filtered_ext
global train_data_len
global np_scaled
global basePred

def load_multivariate_data():

    #----------------------------------Loading-------------------------------------
    global rawData
    #Loading data from file if it exists, else load new data.
    if os.path.isfile("datafile"):
        print("Did find data")
        rawData = pd.read_csv("datafile")
        rawData = yf.download(company, start=startDate, end=endDate)

    else:
        print("Didn't find data")
        rawData = yf.download(company, start=startDate, end=endDate)
        
        toStore = pd.DataFrame(rawData)
        #Store Data      
        if dataStore:            
            toStore.to_csv("datafile")

    df = rawData
    df.dropna(inplace=True)
    # Drop Empty Data

    #Ensure data is sorted by date.
    trainData = df.sort_values(by=['Date']).copy()

    #Filter dataset to only contain selected feature columns.
    data = pd.DataFrame(trainData)
    dataFilter = data[FEATURES]
    
    #Add columns for later prediction.
    dataWithPredict = dataFilter.copy()
    dataWithPredict['Prediction'] = dataWithPredict['Close'] #Copying close values as a placeholder
    global data_filtered_ext
    data_filtered_ext = dataWithPredict

    #----------------------------------Scaling-------------------------------------

    #Get the row count in dataset.
    rowCount = dataFilter.shape[0]

    #Converting to Numpy Array
    npDataUnscaled = np.array(dataFilter)
    npData = np.reshape(npDataUnscaled, (rowCount,-1)) #Reshape based on feature count
    print(npData.shape)

    #Scale data between range of 0-1
    scaler = MinMaxScaler(feature_range=(scaleMin,scaleMax))
    npDataScaled = scaler.fit_transform(npDataUnscaled)


    global normal
    global np_scaled
    normal = scaler
    np_scaled = npDataScaled

    #Save scaler to file for later use.
    dump(scaler, open('scaler.pkl', 'wb'))
    # 'wb' opens for writing to binary

    #Creating Prediction Column Scaler
    scalerPredict = MinMaxScaler(feature_range=(scaleMin,scaleMax))
    dataClose = pd.DataFrame(dataWithPredict['Close'])
    dataCloseScaled  = scalerPredict.fit_transform(dataClose)
    global pred
    pred = scalerPredict

    #Save prediction scaler to file for later use.
    dump(scalerPredict, open('predictionScaler.pkl', 'wb'))
    # 'wb' opens for writing to binary
    

    #Pass data into model building function.
    build_model(data,npDataScaled)


def build_model(data, npDataScaled):

    #----------------------------------Splitting/Shaping Data---------------------------------
    #index for prediction column.
    closeIndex = data.columns.get_loc('Close')

    #Splitting into train/test datasets
    trainDataLength = math.ceil(npDataScaled.shape[0]* splitRatio)
    global train_data_len
    train_data_len = trainDataLength #Storing globally to use later for graphing
    trainData = npDataScaled[0:trainDataLength, :]
    testData = npDataScaled[trainDataLength - predictionDays:, :]

    #Generate the training/test Data
    x_train,y_train = dataSplit(trainData,closeIndex)
    x_test,y_test = dataSplit(testData,closeIndex)

    # Print the shapes: the result is: (rows, training_sequence, features) (prediction value, )
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    #----------------------------------Training Data-------------------------------------------
    model = Sequential()
    print(cellUnits, x_train.shape[1], x_train.shape[2])
    for i in range(layerNumber):
        if i == 0:
            # Runs on the first layer of the model, uses trained data and represents the
            # input layer for model as we give it the previously found input shape.
            model.add(modelType(units=cellUnits, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        elif i == layerNumber - 1:
            # Runs on the last layer of the model. As the next layer will be dense return_sequences
            # must be set to false as Dense layers do not require a three-dimensional sequence.
            model.add(modelType(units=cellUnits, return_sequences=False))
        else:
            # Hidden layers
            model.add(modelType(units=cellUnits, return_sequences=True))

        # Add dropout following each layer, helps prevent overfitting
        # by randomly setting inputs to 0 by frequency of given variable.
        model.add(Dropout(dropoutAmt))

    model.add(Dense(units=denseUnits))
    model.add(Dense(units=outputCount))
    # Add a Dense layer to condense training of previous layers into single dataset for
    # the prediction. Otherwise, would output as many datasets as units given to previous layers.
    model.compile(optimizer='adam', loss='mean_squared_error')
    # sets the algorithm for optimization and its loss function.
    model.fit(x_train, y_train, epochs=epochCount, batch_size=batchCount)
    # finally the model is trained on the separated training data.
    # epoch amount refers to amount of rounds of training, can cause overfitting if overperformed.
    # batch amount refers to amount of input samples trained at a time, reduces computational requirement
    # of training with larger data sets instead of updating the model for every single input.
    loadPredict(model,y_test,x_test,closeIndex)


    


def dataSplit(tData,index):
    #Used in build_model function.
    x, y = [], []
    dataLength = tData.shape[0]
    for i in range(predictionDays,dataLength-outputCount):
        x.append(tData[i-predictionDays:i,:])
        y.append(tData[i:i + outputCount, index])
    
    #Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)
    return x,y
        
def sarimaStackedModel():
    #----------------------------------------Downloading/Splitting Data------------------------------------------------
    dl = yf.download(company, start=startDate, end=endDate)
    df = pd.DataFrame(dl)

    df = df.asfreq('d', method='pad')
    # print(df.index)
    df = df.dropna()

    # print('Shape of data',df.shape)
    # print(df.head())
    # ad_test(df['Close'])

    # Used for testing efficient variable settings of sarima model.
    stepwise_fit = auto_arima(df['Close'], trace=True, suppress_warnings=True,seasonal= True) 
    # print(df.shape)
    train=df.iloc[:-predictionDays]
    test=df.iloc[-predictionDays-1:]
    # print(train.shape,test.shape)
    #----------------------------------------Training Sarima Model---------------------------------------------------
    model=ARIMA(train['Close'],seasonal_order= (seasonalRegression,seasonalIntegrated,seasonalMovingAverage,season))
    #----------------------------------------Fitting Sarima Model---------------------------------------------------
    model2=model.fit()

    # print(model.summary())
    start=len(train)
    end=len(train)+len(test)
    #----------------------------------------Sarima Predictions-----------------------------------------------------
    pred = model2.predict(start,end)
    futurepred = model2.predict(end,end + outputCount - 1) 
    #----------------------------------------Create Ensemble Predictions---------------------------------------------------
    model2 = load_multivariate_data() # Original model v.04 code.
    estimator_list = [
    ('SARIMA',model), ('Other', model2) ]
    stack_model = StackingClassifier(
    estimators=estimator_list, final_estimator=LogisticRegression())
    stack_model = model.fit()
    stackPred = stack_model.predict(start,end)
    
    global basePred
    basePred.index = futurepred.index
    #----------------------------------------Stack Models, Print Esemble Data---------------------------------------
    combinedData = basePred.join(futurepred)
    combinedData = combinedData.set_axis(['OtherR','SarimaR'], axis=1)
    stackPred.head()
    combinedData["OtherW"] = combinedData.OtherR * (1-weight)
    combinedData["SarimaW"] = combinedData.SarimaR * (weight)
    combinedData["Ensemble"] = combinedData.OtherW + combinedData.SarimaW
    print(combinedData)
    #----------------------------------------Plotting Results------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(16, 8))
    plt.plot(test['Close'],color="lightblue", label=f"Testing {company} Data")
    plt.plot(train['Close'],color="black", label=f"Training {company} Data")

    plt.plot(pred, color="orange", label=f"Predicted {company} Price")
    plt.plot(combinedData["Ensemble"], color="red", label=f"Ensemble Prediction {company} Price")

    dl = yf.download(company, start=endDate, end=predictDate)
    plt.plot(dl['Close'], color="blue", label=f"Actual Future {company} Price")

    plt.xlabel("Date")
    plt.ylabel(f"{company} Share Price")
    plt.title(f"Ensemble Model of {modelString} and SARIMA.")
    plt.legend()
    plt.show()


def ad_test(dataset):
     dftest = adfuller(dataset, autolag = 'AIC')
     print("1. ADF : ",dftest[0])
     print("2. P-Value : ", dftest[1])
     print("3. Num Of Lags : ", dftest[2])
     print("4. Num Of Observations Used For ADF Regression:",      dftest[3])
     print("5. Critical Values :")
     for key, val in dftest[4].items():
         print("\t",key, ": ", val)    







def loadPredict(model,yTest,xTest,closeIndex):
#----------------------------------------------Load Initial Data------------------------------------------
    global rawData
#----------------------------------------------Prediction-------------------------------------------------

#Create fresh dataset with same prediction days and filters as trained data

    # Predict Values
    yPredScaled = model.predict(xTest)

    global pred
    # Unscale predictions
    yPred = pred.inverse_transform(yPredScaled)

    yTestUnscaled = pred.inverse_transform(yTest).reshape(-1,outputCount)

    # Prepare Df for multi forecasting
    x_test_unscaled_df, y_pred_df, y_test_unscaled_df = prepare_df(0, xTest, yTest, yPred,closeIndex)

    # Predict Batch
    x_test_latest_batch = np_scaled[-50:,:].reshape(1,50,5)

    # Predict on the batch
    y_pred_scaled = model.predict(x_test_latest_batch)

    y_pred_unscaled = pred.inverse_transform(y_pred_scaled)

    # Prepare the data and plot the input data and the predictions
    x_test_unscaled_df, y_test_unscaled_df, _ = prepare_df(0, x_test_latest_batch, '', y_pred_unscaled,closeIndex)
    plot_multi_test_forecast(x_test_unscaled_df, '', y_test_unscaled_df, "Multivariate-Multistep Prediction")



def prepare_df(i, x, y, y_pred_unscaled,index_close):
    # Undo the scaling on x, reshape the testset into a one-dimensional array, so that it fits to the pred scaler
    global pred
    x_test_unscaled_df = pd.DataFrame(pred.inverse_transform((x[i]))[:,index_close]).rename(columns={0:'XTest'})
    
    y_test_unscaled_df = []
    # Undo the scaling on y
    if type(y) == np.ndarray:
        y_test_unscaled_df = pd.DataFrame(pred.inverse_transform(y)[i]).rename(columns={0:'YTest'})

    # Create a dataframe for the y_pred at position i, y_pred is already unscaled
    y_pred_df = pd.DataFrame(y_pred_unscaled[i]).rename(columns={0:'Prediction'})
    return x_test_unscaled_df, y_pred_df, y_test_unscaled_df

def plot_multi_test_forecast(x_test_unscaled_df, y_test_unscaled_df, y_pred_df, title): 
    # Place y_pred_unscaled and y_test_unscaled into a dataframe with columns pred and true   
    if type(y_test_unscaled_df) == pd.core.frame.DataFrame:
        df_merge = y_pred_df.join(y_test_unscaled_df, how='left')
    else:
        df_merge = y_pred_df.copy()
    
    # Merge the dataframes 
    df_merge_ = pd.concat([x_test_unscaled_df, df_merge]).reset_index(drop=True)
    
    # Plot the linecharts
    """
    plt.plot(df_merge_,color="blue", label=f"Merge2")
    plt.plot(df_merge,color="black", label=f"Merge1")
    plt.plot(x_test_unscaled_df, color="green", label=f"xTestUnscaled")
    plt.plot(y_pred_df, color="orange", label=f"yPredDf")
    plt.legend()
    plt.show()
    """
    global basePred 
    basePred = df_merge
    




def chartData(data,chartPredScaled):
    if displayType == "graph":
        global data_filtered_ext
        global train_data_len

        # The date from which on the date is displayed
        display_start_date = "2019-01-01" 

        # Add the difference between the valid and predicted prices
        train = pd.DataFrame(data_filtered_ext['Close'][:train_data_len + 1]).rename(columns={'Close': 'y_train'})
        valid = pd.DataFrame(data_filtered_ext['Close'][train_data_len:]).rename(columns={'Close': 'y_test'})
        valid.insert(1, "y_pred", chartPredScaled, True)
        valid.insert(1, "residuals", valid["y_pred"] - valid["y_test"], True)
        df_union = pd.concat([train, valid])

        # Zoom in to a closer timeframe
        df_union_zoom = df_union[df_union.index > display_start_date]

        # Create the lineplot
        fig, ax1 = plt.subplots(figsize=(16, 8))
        plt.title("Prediction vs Test")
        plt.ylabel(company, fontsize=18)
        sns.set_palette(["#090364", "#1960EF", "#EF5919"])
        sns.lineplot(data=df_union_zoom[['y_pred', 'y_train', 'y_test']], linewidth=1.0, dashes=False, ax=ax1)
        plt.show()


    if displayType == "candle":
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
        plt.ylabel(f"{company} Share Price")
        plt.title("Candlestick Chart")
        # updating labels and title of chart

        plt.show()


    if displayType == "boxplot":
        chart_small = data[-displayTradingDays:]  # set number of days to display

        plt.figure(figsize=(10, 15))
        xvalues = ["Open", "Close", "Low", "High"]
        plot = plt.boxplot(data[xvalues], patch_artist=True)
        # Open, Close, Low, High, Volume

        plt.yticks(range(100, 500, 20), ["{} $".format(v) for v in range(100, 500, 20)])
        plt.xticks((1, 2, 3, 4), xvalues)

        plt.ylabel(f"{company} Share Price")

        plt.title(f"Boxplot Chart for last {displayTradingDays} days.")
        plt.subplots_adjust(bottom=0.25)  # Make sure dates don't get cutoff screen
        # update labels, layout and title

        colours = ['g', 'b', 'orange', 'r']
        for patch, color in zip(plot['boxes'], colours):
            patch.set_facecolor(color)

        plt.grid(color='grey')

        plt.show()

sarimaStackedModel()
