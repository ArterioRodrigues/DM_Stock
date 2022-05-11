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
df = msft.history(start ='2000-01-01')
df.to_csv('../CSV/msft.csv')

df = pd.read_csv('../CSV/msft.csv')

#data set for training
data = df.filter(['Close'])
dataset = data.values

#scales data into 1-0
scaler = MinMaxScaler(feature_range=(0,1))
scaler_data = scaler.fit_transform(dataset)

#training sets 
train_size = int(df.shape[0] * 0.8)
train_set = scaler_data[0: train_size, :]

x_train = []
y_train = []

for i in range(60, train_size):
    x_train.append(train_set[i-60: i, 0])
    y_train.append(train_set[i,0])



x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape = (x_train.shape[1],1)))
model.add(tf.keras.layers.LSTM(50, return_sequences=False))
model.add(tf.keras.layers.Dense(25))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size = 1, epochs = 1)

test_data = scaler_data[train_size -60:, :]

x_test = []
y_test = dataset[train_size:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])


x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean( predictions - y_test)**2)

print(rmse)

train = data[:train_size]
valid = data[train_size:]
valid['Predictions'] = predictions

#Visualize
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price USD ($)', fontsize = 18)
plt.plot(train["Close"])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predicitons'], loc = 'lower right')
plt.show()
plt.savefig("../plots/visualize.png")

#Get the quote 

apple = yf.Ticker("AAPL")
df = apple.history(start ='2012-01-01', end ='2019-12-17')

new_df = df.filter(['Close'])
last_60_days = new_df[-60:].values
last_120_days = new_df[-120:-60].values

last_60_days_scaled = scaler.transform(last_60_days)
last_120_days_scaled = scaler.transform(last_120_days)

x_test = []
x_test.append(last_60_days_scaled)
x_test.append(last_120_days_scaled)

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

pred_price = model.predict(x_test)

pred_price = scaler.inverse_transform(pred_price)

print(pred_price)

df = apple.history(start ='2019-12-18', end = '2019-12-19')

print(df['Close'])



