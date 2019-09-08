# -*- coding: utf-8 -*-
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from pandas import read_pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import repository as rp
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Concatenate, RepeatVector, RepeatVector, Reshape
from keras.initializers import glorot_normal
from keras.optimizers import Adam
from keras.regularizers import L1L2
import sys
#giving the option of coosing time gap
timegap = int(sys.argv[1])
#read the data set
dataset = read_pickle("Summer_2019_5min.pkl")
#subsample it to  30min
dataset = dataset[dataset.index.minute%timegap==0]

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
if timegap==5:
	minimum_seq_length=288
else:
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
    trains_y[i] = rp.outputreshaper(trains_y[i], outputsequence,outputfeatures)#(samplesize,6,1)
for i in range(len(tests_X)):
    tests_X[i] = rp.inputreshaper(tests_X[i],halfhours, outputsequence)#(samplesize,1,4)
    tests_y[i] = rp.outputreshaper(tests_y[i], outputsequence,outputfeatures)#(samplesize,6,1)


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
batch_size = 1
input_layer = Input(batch_shape=(batch_size,halfhours,inputfeatures), name='input_layer')
reshape_layer = Reshape((halfhours*inputfeatures,),name='reshape_layer')(input_layer)
num_op = outputsequence # ie we want to predict only 1 output
repeater = RepeatVector(num_op, name='repeater')(reshape_layer)
LSTM_layer = LSTM(5, name='LSTM_layer', return_sequences=True, stateful=True, recurrent_regularizer=regularizers)(repeater)
output_layer = Dense(1, name='output_layer')(LSTM_layer)
# #output_layer = TimeDistributed(Dense(outputsequence, kernel_initializer='glorot_normal',  activation='relu'), name='output_layer')(LSTM_layer)
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='mae', optimizer='adam')

# fit network backpropagating the sequences batch wise
noepochs=150
for i in range(len(trains_X)):
    model.fit(np.array(trains_X[i]), np.array(trains_y[i]), epochs=noepochs, batch_size=batch_size,
     validation_data=(tests_X[i], tests_y[i]) , verbose=2, shuffle=False)
    model.reset_states()

#saving the model
#rp.save_model(model)

##testing with the test and entire dataset prediction##############################################
train_plot = []#each element has (samplesize, timestep=outputsequence=6, feature=1)
test_plot = []#each element has (samplesize, timestep=outputsequence=6, feature=1)
model.reset_states()
for i in range(len(trains_X)):
    train_plot.append(model.predict(np.array(trains_X[i]), batch_size=batch_size))
    test_plot.append(model.predict(np.array(tests_X[i]), batch_size=batch_size))
    model.reset_states()

"""
Calculating error across each future timestep in line with the following post
https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/
"""

#concatenating blocks of samples on acis =0 for calclating error along time axis=1
train_pred = np.concatenate(train_plot,axis=0)
train_target= np.concatenate(trains_y,axis=0)
test_pred = np.concatenate(test_plot,axis=0)
test_target = np.concatenate(tests_y,axis=0)


#generalized calc for 3d arrays
for i in range(outputfeatures):
	for j in range(outputsequence):

		#Calculate train error
		rmse = sqrt(mean_squared_error(train_pred[:,j,i],train_target[:,j,i]))
		cvrmse = 100*(rmse/np.mean(train_target[:,j,i]))
		mae = mean_absolute_error(train_pred[:,j,i],train_target[:,j,i])
		file=  open(str(timegap)+'min Results_File.txt', 'a')
		file.write('Time Step {}: Train RMSE={} |Train CVRMSE={} |Train MAE={} \n'.format(j+1,rmse,cvrmse,mae))
		file.close()

		#Calculate test error
		rmse = sqrt(mean_squared_error(test_pred[:,j,i],test_target[:,j,i]))
		cvrmse = 100*(rmse/np.mean(test_target[:,j,i]))
		mae = mean_absolute_error(test_pred[:,j,i],test_target[:,j,i])
		file=  open(str(timegap)+'min Results_File.txt', 'a')
		file.write('Time Step {}: Test RMSE={} |Test CVRMSE={} |Test MAE={} \n'.format(j+1,rmse,cvrmse,mae))
		file.close()

#Plotting the pred versus target curve:train
for i in range(len(train_plot)):
    matplotlib.rcParams['figure.figsize'] = [20.0, 42.0]
    fig, axs = plt.subplots(outputsequence+1)
    for j in range(outputsequence):
        #plot predicted
        axs[j].plot(train_plot[i][:,j,0],'ro-',label='Predicted Energy')
        #plot target
        axs[j].plot(trains_y[i][:,j,0],'go-',label='Actual Energy')
        #Plot Properties
        axs[j].set_title('t+'+str(j+1)+'time step')
        axs[j].set_xlabel('Time points at {} mins'.format(timegap))
        axs[j].set_ylabel('Normalized Energy')
        axs[j].grid(which='both',alpha=100)
        axs[j].legend()
        axs[j].minorticks_on()
    axs[j+1].plot(train_plot[i][:,0,0],'ro-',label='Predicted Energy t+1')
    axs[j+1].plot(train_plot[i][:,1,0],'g*-',label='Predicted Energy t+2')
    axs[j+1].plot(train_plot[i][:,2,0],'bd-',label='Predicted Energy t+3')
    axs[j+1].plot(train_plot[i][:,3,0],'mo-',label='Predicted Energy t+4')
    axs[j+1].plot(train_plot[i][:,4,0],'cd-',label='Predicted Energy t+5')
    axs[j+1].plot(train_plot[i][:,5,0],'k*-',label='Predicted Energy t+6')
    axs[j+1].set_title('Energy at {} minutes differences'.format(timegap))
    axs[j+1].set_xlabel('Time points at {} mins'.format(timegap))
    axs[j+1].set_ylabel('Predicted Normalized Energy')
    axs[j+1].grid(which='both',alpha=100)
    axs[j+1].legend()
    axs[j+1].minorticks_on()

    fig.savefig(str(timegap)+'minutes Train Energy Comparison on Sequence'+str(i+1)+'.pdf',bbox_inches='tight')

#Plotting the pred versus target curve":test
for i in range(len(test_plot)):
    matplotlib.rcParams['figure.figsize'] = [20.0, 42.0]
    fig, axs = plt.subplots(outputsequence+1)
    for j in range(outputsequence):
        #plot predicted
        axs[j].plot(test_plot[i][:,j,0],'ro-',label='Predicted Energy')
        #plot target
        axs[j].plot(tests_y[i][:,j,0],'go-',label='Actual Energy')
        #Plot Properties
        axs[j].set_title('t+'+str(j+1)+'time step')
        axs[j].set_xlabel('Time points at {} mins'.format(timegap))
        axs[j].set_ylabel('Normalized Energy')
        axs[j].grid(which='both',alpha=100)
        axs[j].legend()
        axs[j].minorticks_on()
    axs[j+1].plot(test_plot[i][:,0,0],'ro-',label='Predicted Energy t+1')
    axs[j+1].plot(test_plot[i][:,1,0],'g*-',label='Predicted Energy t+2')
    axs[j+1].plot(test_plot[i][:,2,0],'bd-',label='Predicted Energy t+3')
    axs[j+1].plot(test_plot[i][:,3,0],'mo-',label='Predicted Energy t+4')
    axs[j+1].plot(test_plot[i][:,4,0],'cd-',label='Predicted Energy t+5')
    axs[j+1].plot(test_plot[i][:,5,0],'k*-',label='Predicted Energy t+6')
    axs[j+1].set_title('Energy at {} minutes differences'.format(timegap))
    axs[j+1].set_xlabel('Time points at {} mins'.format(timegap))
    axs[j+1].set_ylabel('Predicted Normalized Energy')
    axs[j+1].grid(which='both',alpha=100)
    axs[j+1].legend()
    axs[j+1].minorticks_on()

    fig.savefig(str(timegap)+'minutes Test Energy Comparison on Sequence'+str(i+1)+'.pdf',bbox_inches='tight')



# pyplot.figure()
# pyplot.plot(all_y_continuous, 'b*-',label='original')
# pyplot.plot(all_pred_continuous,'r+-', label='predicted')
# pyplot.ylabel('BTUs')
# pyplot.xlabel('Samples')
# pyplot.legend()
# pyplot.savefig('Test Preds.pdf', bbox_inches='tight')
