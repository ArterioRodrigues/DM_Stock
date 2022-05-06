import functions as f

ticker = input('Enter a ticker: ')
start_time = input('Enter a starting period: ')
size = float(input('Enter a train size: '))
test_case_size = int(input('Enter a test case size: '))


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




