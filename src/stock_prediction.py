import functions as f
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt

ticker = 'MSFT'
start_time = '2000-01-01'
size = 1
test_case_size = 60


print()

#Get the data
data, dataset= f.ticker_data(ticker, start_time)
train_size = int(dataset.shape[0] * size)

#Build the model
y_test, x_test, scaler, model = f.build_model(dataset, train_size, test_case_size)

#Prediciton
predictions = f.model_predictions(y_test, x_test, scaler, model)

f.Visualize(data, predictions, train_size)


#Get the quote 

apple = yf.Ticker("AAPL")
df = apple.history(start ='2022-03-01')

new_df = df.filter(['Close'])

last_60_days = new_df[-60:].values
data= scaler.transform(last_60_days)

for i in range(60):
    x_test = []
    x_test.append(data)
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    pred_price = model.predict(x_test)
    pred_price = scaler.inverse_transform(pred_price)
    data = data[i:]
    data.append(pred_price)
    print(pred_price)

df = apple.history(start ='2019-12-18', end = '2019-12-19')

print(df['Close'])


