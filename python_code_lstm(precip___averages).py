import pandas as pd
import numpy as np
import math
from pandas import datetime
from pandas import read_csv
from pandas import Series
import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import Bidirectional
from keras.layers import SimpleRNN
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import LSTM
from math import sqrt

#Turning my time series data into a supervised learning one
def timeseries_to_supervised(data, seq_len,pred_len):
	sequence_length = seq_len
	result = []
	resulty =[]
	#Adjusting the data by specified windowsize
	for index in range(len(data) - sequence_length-pred_len+1):
		result.append(data[index: index + sequence_length])
		resulty.append(data[index+sequence_length:index + sequence_length+pred_len])
	result = np.array(result)
	resulty =np.array(resulty)
	return result, resulty

# Changing the scale of the data to [-1, 1]
def scale(train, test):
	#Changing the scale from -1 to 1
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape)
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape)
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

#inverse scaling for a forecasted value
def invert_scale(scaler, value):
		array = np.array(value)
		array0 = array.reshape(1, len(array))
		inverted = scaler.inverse_transform(array0)
		return array, array0, inverted[0, -1]

#fitting an LSTM network to training data
def fit_lstm(layers,lr_value,ws):
	model = Sequential()
	#Two hidden layers are incorporated in this LSTM algorithm
	model.add(LSTM(input_dim=layers[0],output_dim=layers[1],return_sequences=True))
	model.add(Dropout(0.3))
	model.add(LSTM(layers[2],return_sequences=True))
	model.add(Dropout(0.3))
	model.add(Dense(output_dim=layers[3]))
	model.add(LeakyReLU(alpha=lr_value))
	#I would use a soft sign activation function for the temperature variables
	#model.add(Activation("softsign"))
	model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
	return model

#creating forecasts
def forecast_lstm(model, batch_size, X):
        yhat = model.predict(X, batch_size=batch_size)
        return yhat

data = np.load("C:\\Users\\jeffr\\Google Drive\\ST4.201211_201401.rainy.npy")
avg=data.mean(axis=(1,2))
#avg=data[:,-1,-1]
df = pd.DataFrame(avg, columns=['Precip'])
df2=df.values
#raw_values = df2.values

#splitting the data into training and test sets
train_size = int(len(df2) * 0.80)
test_size = len(df2) - train_size
train, test = df2[0:train_size], df2[train_size:len(df2)]
# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)
# transforming the training and test data to be supervised learning; the window size is set
windowsize=1 #2 is the best choice
supervised_values, y_values = timeseries_to_supervised(train_scaled, windowsize,windowsize)
test_values, test_y_values = timeseries_to_supervised(test_scaled, windowsize,windowsize)

df.plot(legend=None,color='k')
plt.show()

# fitting the model
lstm_model = fit_lstm([1, 50, 50, 1],0.25,windowsize)
history=lstm_model.fit(supervised_values,y_values,batch_size=1,nb_epoch=10,validation_split=.25)
# forecasting the entire training dataset to build up state for forecasting
train_pred=lstm_model.predict(supervised_values, batch_size=1)

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_values)):
        # making multi-step forecast
		X, y = test_values[:, :,:], test_y_values[:,:,: ]
		yhat = forecast_lstm(lstm_model, 1, X)
		#reshaping the data back into a single list of values
		yhat_new=yhat[0::windowsize]
		yhat_new=yhat_new.reshape((len(yhat_new))*windowsize,1) #yhat_new.shape[0]*3
		X_new=X[0::windowsize]
		X_new=X_new.reshape((len(X_new))*windowsize,1) #X_new.shape[0]*3
		newvalue=[X_new[i], yhat_new[i]]
		# invert scaling
		thearray, thearray2, yhat_new0 = invert_scale(scaler, newvalue)
# storing forecasts
		predictions.append(yhat_new0)


plot_test_values=df2[-test_size:-((2*windowsize)-1)]
# report LSTM model performance
rmse = sqrt(mean_squared_error(plot_test_values, predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
plt.plot(plot_test_values,color='k')
plt.plot(predictions,color='b')
plt.xlabel('Test Data Time Steps')
plt.ylabel('Precipitation Values')
plt.show()

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
