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
#put here the path to the repository.py file
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
minvect=np.ones((dataset.values.shape[0],1))*48.98
maxvect=np.ones((dataset.values.shape[0],1))*91.66
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
    inputArraytemp=outputDFrames[i].iloc[:,1:5].values
    outputArraytemp=outputDFrames[i].iloc[:,0].values
    if i>0:
       inputArrays= np.vstack((inputArrays ,inputArraytemp ))     
       outputArrays= np.hstack((outputArrays ,outputArraytemp ))    
    else:
       inputArrays=  inputArraytemp   
       outputArrays= outputArraytemp
 
#configure shape for each sample
varIndex=np.array([4])#vector containing the index of the output variable, zero-index
delays=np.array([0]) #delay in the output variable
#extract continuous sequences
sequencelength=1
inputArray,outputArray=inputArrays,outputArrays;

#training/testing separation
#sequential splitting
Nosamples=inputArray.shape[0]
proportion=round(3/4*Nosamples)
train_X,test_X =inputArray[0:proportion,:], inputArray[proportion:Nosamples,:]
train_y, test_y = outputArray[0:proportion,], outputArray[proportion:Nosamples,]

#reshaping before training
hours=sequencelength
outputfeatures=1
n_features=5
input_features=n_features+delays-outputfeatures
n_steps=hours*input_features[0]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], hours*input_features[0]))
test_X = test_X.reshape((test_X.shape[0], hours*input_features[0]))

# design network ############################################################
model = Sequential()
model.add(Dense(200, activation='relu', input_dim=n_steps))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
noepochs=150
history = model.fit(train_X, train_y, epochs=noepochs, validation_data=(test_X, test_y), verbose=2)

#saving the model
rp.save_model(model)

# testing network ############################################################

test_pred= model.predict(test_X)
# calculate RMSE testing
rmsetest = sqrt(mean_squared_error(test_y, test_pred))
print('Test RMSE: %.3f' % rmsetest)
print('Test CVRMSE: %.3f' % (rmsetest/np.mean(test_y)))

# plot error distribution
mae=abs(test_y[:,].T- test_pred[:,].T)
print('Test MAE: %.3f' % np.mean(mae))
fig1, ax1 = pyplot.subplots()
ax1.set_title('Basic Plot')
ax1.boxplot(mae.T)

test_pred_continuous=test_pred
test_y_continuous=test_y
test_y_continuoustest=test_y

# plot history
pyplot.figure()
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

#plot predicted and real variables
pyplot.figure()
pyplot.plot(test_y, 'b*-',label='original')
pyplot.plot(test_pred,'r+-', label='predicted')
pyplot.ylabel('BTUs')
pyplot.xlabel('Samples')
pyplot.legend()
pyplot.show()

#entire dataset prediction
totalX=inputArray
totaly=outputArray
total_pred= model.predict(totalX)

test_pred_continuous=total_pred
test_y_continuous=totaly

rmsetrain = sqrt(mean_squared_error(totaly, total_pred))
print('All RMSE: %.3f' % rmsetrain)
print('All CVRMSE: %.3f' % (rmsetest/np.mean(totaly)))
mae=abs(totaly[:,].T- total_pred[:,].T)
print('All MAE: %.3f' % np.mean(mae))

pyplot.figure()
pyplot.plot(totaly, 'b*-',label='original')
pyplot.plot(total_pred,'r+-', label='predicted')
pyplot.ylabel('BTUs')
pyplot.xlabel('Samples')
pyplot.legend()
pyplot.show()
