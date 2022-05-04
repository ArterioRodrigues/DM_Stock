import math
from pickletools import optimize
import pandas as pd
import yfinance as yf
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler




msft = yf.Ticker("MSFT")
df = msft.history(start ='2021-01-01')
df.to_csv('../CSV/msft.csv')

df = pd.read_csv('../CSV/msft.csv')

#data set for training
data = df.filter(['Close'])
dataset = data.values

#scales data into 1-0
scaler = MinMaxScaler(feature_range=(0,1))
scaler_data = scaler.fit_transform(dataset)

#Get the quote 

apple = yf.Ticker("AAPL")
df = apple.history(start ='2012-01-01', end ='2019-12-17')

df = df.filter(['Close'])
day = []
for i in range(10):
    x = scaler.transform(df[-(60+i):-i].values)
    day.append(x)

print(day)
x_test = []

for i in df:
    x_test.append(i)

print(x_test)
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))












def qoute(model, ticker, scaler, num_pred, test_case_size, usr_start = '2021-01-01', usr_end = '2022-01-01'):
    ticker = yf.Ticker(ticker)
    df = ticker.history(start = '2000-01-01')

    df = df.filter(['Close'])
    day = []
    for i in range(num_pred):
        day.append(scaler.transform(df[-test_case_size-i:-i].values))

    x_test = []

    for i in df:
        x_test.append(i)

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    pred_price = model.predict(x_test)
    pred_price = scaler.inverse_transform(pred_price)

    print(pred_price)