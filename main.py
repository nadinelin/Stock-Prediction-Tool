import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import numpy as np
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


'''
Load Data
'''
# Company Selection
company = 'GOOG' 

# Time Frame Selection
start = dt.datetime(2015,1,1)
end = dt.datetime(2020,1,1)
data = web.DataReader(company, 'yahoo', start, end)

'''
Prepare Data
'''
# Normalization
scaler = MinMaxScaler(feature_range = (0,1))

# Stock price when closing
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

# Use the past 70 days data to predict stock trend for the next day
prediction_days = 70 
x_train = []
y_train = []

'''
Prepare Training Data
'''
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days: x, 0])
    y_train.append(scaled_data[x,0])

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

'''
Building LSTM Model 
'''
model = Sequential()
# Level 1
model.add(LSTM(units=60, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
# Level 2
model.add(LSTM(units=60, return_sequences=True))
model.add(Dropout(0.2))
# Level 3
model.add(LSTM(units=60))
model.add(Dropout(0.2))

model.add(Dense(units=1)) 

# Prediction of next closing value
model.compile(optimizer='adam', loss='mean_squared_error')

# Train Model
model.fit(x_train, y_train, epochs=25, batch_size=32)

'''
Accuracy Level Test Using Existing Data
'''

# Load Test Data
test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

test_data = web.DataReader(company,'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

# Make Predictions using Test Data
x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot The Test Predictions
length = len(predicted_prices)
begin = dt.date(2020,6,1)
delta = dt.timedelta(days=1)
d = begin
ticks = []
for i in range(length):
    if i % 50 == 0:
        ticks.append(d.strftime("%Y-%m-%d"))
    d += delta
    
plt.plot(actual_prices, color='blue', label=f"Actual {company} Price")
plt.plot(predicted_prices, color='green', label=f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.ylim(ymin=0)
plt.xticks([i for i in range(length) if i % 50 == 0], ticks)
plt.legend()
plt.show()

# Prediction for Next Day
tmrw_data = [model_inputs[-prediction_days:,0]]
tmrw_data = np.array(tmrw_data)
tmrw_data = np.reshape(tmrw_data, (tmrw_data.shape[0], tmrw_data.shape[1],1))
ans = model.predict(tmrw_data)
ans = scaler.inverse_transform(ans)
print(ans)