# -*- coding: utf-8 -*-
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from pandas import read_pickle
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import model_from_json
from keras.regularizers import L1L2
from math import sqrt
from matplotlib import pyplot
import numpy as np
import sys
sys.path.insert(0, 'D:\\Programming\\research\\LSTM\\Buildings\\Repo')
import repository as rp

#read the data set
dataSet = read_pickle("SummerData30min.pkl")

#remove large outliers
dataset=rp.replace_outliers(dataSet,'CoolE')
dataset=rp.replace_outliers(dataset,'HeatE')

#scale the data min-max option
scaler = MinMaxScaler(feature_range=(0, 1))
first=scaler.fit_transform(dataset.values[:,0:3])
last=scaler.fit_transform(dataset.values[:,4:8])
minvect=np.ones((dataset.values.shape[0],1))*50 #now 50 - before 48.98
maxvect=np.ones((dataset.values.shape[0],1))*80 #now 80 - before 91.66
dischtempscaled=((dataset.values[:,3]-minvect.T)/(maxvect.T - minvect.T)).T
totalvect=np.hstack((first,dischtempscaled,last))
dataset = DataFrame(totalvect, columns=dataset.columns, index=dataset.index)

#Aggregated data - is already aggregated
#dataset['Aggregated'] = dataset.apply(lambda row: row.CoolE + row.HeatE, axis = 1)

#rename first column because its really the Aggregated energy
dataset = dataset.rename(columns={'CoolE': 'Aggregated'})

#drop useless variables
dataset=dataset.drop(['HeatE'], axis=1)

#extract sequences larger than L samples
sequencelength=2 #minimum lenght of outliers admitted
outputDFrames=rp.continuous_sequencesnan(dataset,sequencelength)
minimum_seq_lenght=48 #one day
counteri=0
for i in range(len(outputDFrames)):
    if len(outputDFrames[counteri])<minimum_seq_lenght:
        outputDFrames.pop(counteri)
    else:
        counteri=counteri+1

#interpolate within each dataFrame missing values 
for i in range(len(outputDFrames)): 
    outputDFrames[i]=outputDFrames[i].interpolate(method='linear')

#reorganize data to predict future observation
for i in range(len(outputDFrames)):    
    futureAgg=outputDFrames[i]['Aggregated'].shift(-1)
    outputDFrames[i]['Aggregated']=futureAgg
    outputDFrames[i] = outputDFrames[i][~np.isnan(outputDFrames[i]['Aggregated'])]

#extract input and output variables
inputArrays,outputArrays=[],[]
for i in range(len(outputDFrames)):    
    inputArray=outputDFrames[i].iloc[:,1:5].values
    outputArray=outputDFrames[i].iloc[:,0].values
    inputArrays.append(inputArray)
    outputArrays.append(outputArray)

#divide training and testing the different sequences
trainpercent=3/4;
trains_X,tests_X=[],[]
trains_y,tests_y=[],[]
for i in range(len(outputDFrames)): 
    inputArray=inputArrays[i]
    outputArray=outputArrays[i]
    Notrainingsets=len(inputArray)
    proportion=round(3/4*Notrainingsets)
    train_X=inputArray[0:proportion]
    trains_X.append(train_X)
    test_X=inputArray[proportion:Notrainingsets]
    tests_X.append(test_X)
    train_y=outputArray[0:proportion]
    trains_y.append(train_y)
    test_y=outputArray[proportion:Notrainingsets]
    tests_y.append(test_y)
    
#reshaping before training
hours=1
inputfeatures=4
outputfeatures=1
delays=np.array([1])
input_features=inputfeatures+delays-outputfeatures

# reshape input to be 3D [samples, timesteps, features]
for i in range(len(trains_X)):
    trains_X[i]= trains_X[i].reshape((trains_X[i].shape[0], hours, input_features[0]))
    if trains_y[i].size ==1:
        trains_y[i]=[trains_y[i],] 
for i in range(len(tests_X)):
    tests_X[i]= tests_X[i].reshape((tests_X[i].shape[0], hours, input_features[0]))

for i in trains_X:
    print(i.shape)


# #possible regularization strategies#################################
# #regularizers = L1L2(l1=0.0, l2=0.0) #none
# #regularizers =  L1L2(l1=0.01, l2=0.0) #l1
# #regularizers =  L1L2(l1=0.0, l2=0.001) #l2
# regularizers =  L1L2(l1=0.01, l2=0.001)#l1l2
#
# # design network ############################################################
# batch_size=1
# model = Sequential()
# model.add(LSTM(5, batch_input_shape=(batch_size, trains_X[0].shape[1], trains_X[0].shape[2]), stateful=True, recurrent_regularizer=regularizers))
# model.add(Dense(1))
# model.compile(loss='mae', optimizer='adam')
#
#
# # fit network backpropagating the sequences batch wise
# noepochs=150
# for i in range(len(trains_X)):
#     model.fit(np.array(trains_X[i]), np.array(trains_y[i]), epochs=noepochs, batch_size=batch_size, validation_data=(tests_X[i], tests_y[i]) , verbose=2, shuffle=False)
#     model.reset_states()
#
# ##saving the model
# rp.save_model(model)
#
# ##testing with the test and entire dataset prediction##############################################
# test_pred = []
# yhat,yhattest=[],[]
# y,ytest=[],[]
# model.reset_states()
# for i in range(len(trains_X)):
#     y1= model.predict(np.array(trains_X[i]), batch_size=batch_size)
#     y2= model.predict(np.array(tests_X[i]), batch_size=batch_size)
#     model.reset_states()
#     sequence=np.vstack((y1,y2 ))
#     truesequence=np.hstack(([trains_y[i],] ,[tests_y[i],]  ))
#
#     if i>0:
#         yhat=np.vstack((yhat,sequence ))
#         y=np.hstack((y ,truesequence  ))
#         yhattest=np.vstack((yhattest,y2 ))
#         ytest=np.hstack((ytest ,tests_y[i]  ))
#     else:
#         yhat=sequence
#         y=truesequence
#         yhattest=y2
#         ytest=tests_y[i]
#
# ##### test errors
# test_pred_continuous= yhattest
# test_y_continuous= ytest.T
# all_pred_continuous= yhat
# all_y_continuous= y.T
#
# rmsetest = sqrt(mean_squared_error(test_y_continuous, test_pred_continuous))
# print('Test RMSE: %.3f' % rmsetest)
# print('Test CVRMSE: %.3f' % (rmsetest/np.mean(test_y_continuous)))
#
# mae=abs(test_y_continuous[:,].T- test_pred_continuous[:,].T)
# print('Test MAE: %.3f' % np.mean(mae))
#
# ######### all data set errors
# rmsetest = sqrt(mean_squared_error(all_y_continuous, all_pred_continuous))
# print('All RMSE: %.3f' % rmsetest)
# print('All CVRMSE: %.3f' % (rmsetest/np.mean(all_y_continuous)))
#
# mae=abs(all_y_continuous[:,].T- all_pred_continuous[:,].T)
# print('All MAE: %.3f' % np.mean(mae))
#
# pyplot.figure()
# pyplot.plot(all_y_continuous, 'b*-',label='original')
# pyplot.plot(all_pred_continuous,'r+-', label='predicted')
# pyplot.ylabel('BTUs')
# pyplot.xlabel('Samples')
# pyplot.legend()
# pyplot.show()
#
#
#
#
#
#
