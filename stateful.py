# -*- coding: utf-8 -*-
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from pandas import read_pickle
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import matplotlib
from matplotlib import pyplot
import numpy as np
import repository as rp
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Concatenate, RepeatVector, RepeatVector
from keras.initializers import glorot_normal
from keras.optimizers import Adam


#read the data set
dataset = read_pickle("Summer_2019_5min.pkl")
#subsample it to  30min
dataset = dataset[dataset.index.minute%30==0]

#remove large outliers
#dataset=rp.replace_outliers(dataSet,'CoolE')
#dataset=rp.replace_outliers(dataset,'HeatE')

#scale the data min-max option
scaler = MinMaxScaler(feature_range=(0, 1))
first=scaler.fit_transform(dataset.values[:,0:3])
last=scaler.fit_transform(dataset.values[:,4].reshape(-1,1))
minvect=np.ones((dataset.values.shape[0],1))*50 #now 50 - before 48.98
maxvect=np.ones((dataset.values.shape[0],1))*80 #now 80 - before 91.66
dischtempscaled=((dataset.values[:,3]-minvect.T)/(maxvect.T - minvect.T)).T
totalvect=np.hstack((first,dischtempscaled,last))
dataset = DataFrame(totalvect, columns=dataset.columns, index=dataset.index)

#Aggregated data - is already aggregated
#dataset['Aggregated'] = dataset.apply(lambda row: row.CoolE + row.HeatE, axis = 1)

#rename first column because its really the Aggregated energy
#dataset = dataset.rename(columns={'CoolE': 'Aggregated'})

#drop useless variables
#dataset=dataset.drop(['HeatE'], axis=1)

#extract sequences larger than L samples
outputDFrames=rp.subsequencing(dataset)#rp.continuous_sequencesnan(dataset,sequencelength)
minimum_seq_length=48 #one day
counteri=0
for i in range(len(outputDFrames)):
    if len(outputDFrames[counteri])<minimum_seq_length:
        outputDFrames.pop(counteri)
    else:
        counteri=counteri+1

#interpolate within each dataFrame missing values
for i in range(len(outputDFrames)):
    outputDFrames[i]=outputDFrames[i].interpolate(method='linear')

#reorganize data to predict future observation
for i in range(len(outputDFrames)):
    futureAgg=outputDFrames[i]['TotalE'].shift(-1)
    outputDFrames[i]['TotalE']=futureAgg
    outputDFrames[i] = outputDFrames[i][~np.isnan(outputDFrames[i]['TotalE'])]

#extract input and output variables
inputArrays,outputArrays=[],[]
for i in range(len(outputDFrames)):
    inputArray=outputDFrames[i].iloc[:,:-1].values
    outputArray=outputDFrames[i].iloc[:,-1].values
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
halfhours=1
inputfeatures=4
outputfeatures=1
outputsequence = 6

for i in range(len(trains_X)):
    trains_X[i] = rp.inputreshaper(trains_X[i],halfhours, outputsequence)#(samplesize,1,4)
    trains_y[i] = rp.outputreshaper(trains_y[i], outputsequence)#(samplesize,6)
for i in range(len(tests_X)):
    tests_X[i] = rp.inputreshaper(tests_X[i],halfhours, outputsequence)#(samplesize,1,4)
    tests_y[i] = rp.outputreshaper(tests_y[i], outputsequence)#(samplesize,6)


for i in trains_X:
    print(i.shape)
for i in trains_y:
    print(i.shape)
for i in tests_X:
    print(i.shape)
for i in tests_y:
    print(i.shape)


#possible regularization strategies#################################
regularizers =  L1L2(l1=0.01, l2=0.001)#l1l2

# design network ############################################################
input_layer = Input(batch_shape=(batch_size,halfhours,inputfeatures), name='input_layer')
reshape_layer = Reshape((halfhours*inputfeatures,),name='reshape_layer')(input_layer)
num_op = outputsequence # ie we want to predict only 1 output
repeater = RepeatVector(num_op, name='repeater')(reshape_layer)
LSTM_layer = LSTM(5, name='LSTM_layer', return_sequences=True, stateful=True, recurrent_regularizer=regularizer)(repeater)
output_layer = Dense(1, name='output_layer')(LSTM_layer)
# #output_layer = TimeDistributed(Dense(outputsequence, kernel_initializer='glorot_normal',  activation='relu'), name='output_layer')(LSTM_layer)
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='mae', optimizer='adam')

# fit network backpropagating the sequences batch wise
noepochs=150
for i in range(len(trains_X)):
    model.fit(np.array(trains_X[i]), np.array(trains_y[i]), epochs=noepochs, batch_size=batch_size, validation_data=(tests_X[i], tests_y[i]) , verbose=2, shuffle=False)
    model.reset_states()

##saving the model
rp.save_model(model)

##testing with the test and entire dataset prediction##############################################
test_pred = []
yhat,yhattest=[],[]
y,ytest=[],[]
model.reset_states()
for i in range(len(trains_X)):
    y1= model.predict(np.array(trains_X[i]), batch_size=batch_size)
    y2= model.predict(np.array(tests_X[i]), batch_size=batch_size)
    model.reset_states()
    sequence=np.vstack((y1,y2 ))
    truesequence=np.hstack(([trains_y[i],] ,[tests_y[i],]  ))

    if i>0:
        yhat=np.vstack((yhat,sequence ))
        y=np.hstack((y ,truesequence  ))
        yhattest=np.vstack((yhattest,y2 ))
        ytest=np.hstack((ytest ,tests_y[i]  ))
    else:
        yhat=sequence
        y=truesequence
        yhattest=y2
        ytest=tests_y[i]

##### test errors
test_pred_continuous= yhattest
test_y_continuous= ytest.T
all_pred_continuous= yhat
all_y_continuous= y.T

rmsetest = sqrt(mean_squared_error(test_y_continuous, test_pred_continuous))
print('Test RMSE: %.3f' % rmsetest)
print('Test CVRMSE: %.3f' % (rmsetest/np.mean(test_y_continuous)))

mae=abs(test_y_continuous[:,].T- test_pred_continuous[:,].T)
print('Test MAE: %.3f' % np.mean(mae))

######### all data set errors
rmsetest = sqrt(mean_squared_error(all_y_continuous, all_pred_continuous))
print('All RMSE: %.3f' % rmsetest)
print('All CVRMSE: %.3f' % (rmsetest/np.mean(all_y_continuous)))

mae=abs(all_y_continuous[:,].T- all_pred_continuous[:,].T)
print('All MAE: %.3f' % np.mean(mae))

pyplot.figure()
pyplot.plot(all_y_continuous, 'b*-',label='original')
pyplot.plot(all_pred_continuous,'r+-', label='predicted')
pyplot.ylabel('BTUs')
pyplot.xlabel('Samples')
pyplot.legend()
pyplot.savefig('Test Preds.pdf', bbox_inches='tight')
