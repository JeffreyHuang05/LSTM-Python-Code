import pandas as pd
import numpy
import math
from pandas import datetime
from pandas import read_csv
from pandas import Series
import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import Bidirectional
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import LSTM
from math import sqrt

import tensorflow as tf

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

#Turning my time series data into a supervised learning one
def timeseries_to_supervised(data, seq_len,pred_len):
	sequence_length = seq_len
	result = []
	resulty =[]
	for index in range(len(data) - sequence_length-pred_len+1):
		result.append(data[index: index + sequence_length])
		resulty.append(data[index+sequence_length:index + sequence_length+pred_len])
	result = numpy.array(result) 
	resulty =numpy.array(resulty)
	return result, resulty

# Changing the scale of the data to [-1, 1]
def scale(train, test):
	scaler = MinMaxScaler(feature_range=(-1, 1))
	# transform train
	scaler = scaler.fit(train)
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

#inverse scaling for a forecasted value
def invert_scale(scaler, value):
		array = numpy.array(value)
		array0 = array.reshape(1, len(array))
		inverted = scaler.inverse_transform(array0)
		return array, array0, inverted[0, -1]

#fitting an LSTM network to training data
def fit_lstm(layers,lr_value,ws):
	model = Sequential()
	#Two hidden layers are being used for this LSTM algorithm
	model.add(LSTM(input_dim=layers[0],output_dim=layers[1],return_sequences=True))
	model.add(Dropout(0.5))
	model.add(LSTM(layers[2],return_sequences=True))
	model.add(Dropout(0.5))
	model.add(Dense(units=layers[3]))
	model.add(LeakyReLU(alpha=lr_value))
	model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
	return model

#creating forecasts
def forecast_lstm(model, batch_size, X):
        yhat = model.predict(X, batch_size=batch_size)
        return yhat

def experiment(repeats, series,neurons,batchsize=6,windowsize=2,epochs=80):
	#splitting the data into training and test sets
	train_size = int(len(series) * 0.80)
	test_size = int(len(series) - train_size)
	train, test = series[0:train_size,:], series[train_size:len(series),:]    
# transform the scale of the data
	scaler, train_scaled, test_scaled = scale(train, test)
	# transform data to be supervised learning
	supervised_values, y_values = timeseries_to_supervised(train_scaled, windowsize,windowsize)
	test_values, test_y_values = timeseries_to_supervised(test_scaled, windowsize,windowsize)   
	error_scores = list()
	for r in range(repeats):
# fitting the model
		lstm_model = fit_lstm(neurons,0.5,windowsize)
		history=lstm_model.fit(supervised_values,y_values,batch_size=batchsize,nb_epoch=epochs,validation_split=.25)
# forecasting the entire training dataset to build up state for forecasting
		train_pred=lstm_model.predict(supervised_values, batch_size=batchsize)
# walk-forward validation on the test data
		predictions = list()
		for i in range(len(test_values)):#
        # make multi-step forecast
			X, y = test_values[:, :,:], test_y_values[:,:,: ]           
			yhat = forecast_lstm(lstm_model, batchsize, X)          
			#reshaping the data back into a single list of values
			yhat_new=yhat[0::windowsize]
			yhat_new=yhat_new.reshape((len(yhat_new))*windowsize,1)   
			X_new=X[0::windowsize]
			X_new=X_new.reshape((len(X_new))*windowsize,1) 
			newvalue=[X_new[i], yhat_new[i]]
			# invert scaling
			thearray, thearray2, yhat_new0 = invert_scale(scaler, newvalue)
	# store forecast
			predictions.append(yhat_new0)        
# report performance and plotting the results
		rmse = sqrt(mean_squared_error(test[:-((2*windowsize)-1)], predictions))
		print('Test RMSE: %.3f' % rmse)
		#plt.plot(test[:-((2*windowsize)-1)],color='k')
		#plt.plot(predictions,color='darkgray')
		#plt.xlabel('Test Data Time Steps')
		#plt.ylabel('Precipitation Values')     
		#plt.legend(['test data', 'predictions'], loc='upper right')
		#plt.savefig('lstm_predictions_az.png')
		#plt.show()
		# list all data in history
		#print(history.history.keys())
		# summarize history for accuracy
		#plt.plot(history.history['acc'])
		#plt.plot(history.history['val_acc'])
		#plt.title('model accuracy')
		#plt.ylabel('accuracy')
		#plt.xlabel('epoch')
		#plt.legend(['train', 'test'], loc='upper left')
		#plt.show()
		plt.plot(history.history['loss'],color='k')
		plt.plot(history.history['val_loss'],color='darkgray')
		plt.title('Colorado')
		#plt.ylabel('Loss')
		plt.xlabel('Epochs')
		#plt.legend(['train', 'test'], loc='upper right')
		plt.savefig('loss_function_co.png')
		plt.show()
		error_scores.append(rmse)
	return error_scores

#Creating and setting up the data set
data = numpy.load("C:\\Users\\jeffr\\OneDrive\\Documents\\the data\\ST4.201411_201502_co.npy")
print(data.shape)
data1=data[:,1,8]
df = pd.DataFrame(data1, columns=['Precip'])
df2=df.values
repeats = 1