from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from pandas import DateOffset
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import model_from_json
from math import sqrt
from matplotlib import pyplot
import numpy as np
import pandas as pd
import pickle
 
# parse dates
def parser(x):
    return datetime.strptime(x, '%m/%d/%Y %H:%M')
# replace outliers with nan
def replace_outliers(df,varToremove): 
    meanTemp = [] 
    stdTemp= [] 
    for x in range(24):
        hour=str(x)+':00' 
        dfhour=df.between_time(hour,hour)
        meanTemp.append(dfhour[varToremove].mean())
        stdTemp.append(dfhour[varToremove].std())
    dfnew = DataFrame(data=None, columns=df.columns, index=None)
    for i in range(df.shape[0]):  
        observation=df.iloc[i]
        houri=df.index[i].hour
        if (abs(observation[varToremove]-meanTemp[houri]) < 3*stdTemp[houri]):
             dfnew=dfnew.append(observation)  
        else:
            observation[varToremove]=None
            dfnew=dfnew.append(observation)      
    return dfnew

# remove hourly outliers
def remove_hour_outliers(df,varToremove): 
    meanTemp = [] 
    stdTemp= [] 
    for x in range(24):
        hour=str(x)+':00' 
        dfhour=df.between_time(hour,hour)
        meanTemp.append(dfhour[varToremove].mean())
        stdTemp.append(dfhour[varToremove].std())
    dfnew = DataFrame(data=None, columns=df.columns, index=None)
    for i in range(df.shape[0]):  
        observation=df.iloc[i]
        houri=df.index[i].hour
        if (abs(observation[varToremove]-meanTemp[houri]) < 3*stdTemp[houri]):
             dfnew=dfnew.append(observation)           
    return dfnew

# set the shape for steps back in time 
def configure_steps(df,varIndex,delays): 
    cols = df.columns.tolist();
    for i in range(varIndex.shape[0]):
        variableName=cols[varIndex[i]]        
        for j in range(delays.shape[0]):
            for k in range(delays[j]):
                variableNewName=variableName+'-'+str(k+1)
                df[variableNewName] = df[variableName].shift(k+1)
                df=df.drop(df.index[0])
                colstemp = df.columns.tolist();
                cols2 = [colstemp[-1]]+colstemp[:-1]
                df=df.reindex(columns=cols2)
#    df=df.drop(df.index[0])
    return df

#extract continuous sequences in different dataframes
def continuous_sequences(df,sequencelength, separation_interval): 
    counter=0    
    hourArray= []
    dataframeslist=[]
    dfnew = DataFrame(data=None, columns=df.columns, index=None)                
    for i in range(df.shape[0]):
        hourArray.append(df.index[i].hour);
        if i>0:
            if (df.index[i] -df.index[i-1]).seconds<=separation_interval:
                counter=counter+1
            else:
                if not dfnew.empty:
                    dataframeslist.append(dfnew)                    
                dfnew = DataFrame(data=None, columns=df.columns, index=None)
                counter=0  
                
            if counter==sequencelength-1:                 
                dfnew=dfnew.append(df.iloc[i-sequencelength+1:i+1,])
            elif counter>sequencelength-1:
                dfnew=dfnew.append(df.iloc[i,])                
    return dataframeslist

#extract continuous sequences in different dataframes where nan are missing values
def continuous_sequencesnan(df,sequencelength): 
    counter=0    
    hourArray= []
    dataframeslist=[]
    dfnew = DataFrame(data=None, columns=df.columns, index=None)                
    for i in range(df.shape[0]):
        if (np.isnan(df.iloc[i]).any()):
            counter=counter+1  
        else:
            if counter !=0:
                if counter>=sequencelength:
                    if not dfnew.empty:
                        dataframeslist.append(dfnew)                    
                        dfnew = DataFrame(data=None, columns=df.columns, index=None)
                        dfnew=dfnew.append(df.iloc[i,]) 
                else:
                    dfnew=dfnew.append(df.iloc[i-counter:i,])          
                counter=0
            else:
                dfnew=dfnew.append(df.iloc[i,])                    
    dataframeslist.append(dfnew)                        
    return dataframeslist

#extract continuous overlapping sequences many to one and set the structure of each sequence
def continuous_overlapping_sequences(df,sequencelength,varIndex,delays): 
    if sequencelength<max(delays)+1:
        return "The length of the sequence must be larger at least in one than the delays"
    hourArray= []
    inputArray= None
    outputArray= None  
    counter=0    
    for i in range(df.shape[0]):
        hourArray.append(df.index[i].hour);
        if i>0:
            if (df.index[i] -df.index[i-1]).seconds==3600 and (df.index[i] -df.index[i-1]).days==0:
                counter=counter+1
            else:
                counter=0  
            if counter>=sequencelength-1: 
                dfnew = DataFrame(data=None, columns=df.columns, index=None)
                dfnew=dfnew.append(df.iloc[i-sequencelength+1:i+1,])
                #now reshape each sample of this sequence
                dfnew=configure_steps(dfnew,varIndex,delays)
                dfvalue=dfnew.values
                X, y = dfvalue[:, 0:dfvalue.shape[1]-1], dfvalue[:,  -1]
                inputSeq = X.reshape(1,(sequencelength-delays[0])*(X.shape[1]))
                outputSeq=y[-1]
                if inputArray is None:
                    inputArray=np.array(inputSeq)
                    outputArray=np.array(outputSeq)
                else:
                    inputArray=np.concatenate((inputArray,inputSeq))
                    outputArray=np.hstack((outputArray,outputSeq))
    return inputArray,outputArray

#extract continuous non-overlapping sequences many to one and set the structure of each sequence
def continuous_nonoverlapping_sequences(df,sequencelength,varIndex,delays): 
    if sequencelength<max(delays)+1:
        return "The length of the sequence must be larger at least in one than the delays"
    hourArray= []
    inputArray= None
    outputArray= None  
    counter=0    
    for i in range(df.shape[0]):
        hourArray.append(df.index[i].hour);
        if i>0:
            if (df.index[i] -df.index[i-1]).seconds==3600 and (df.index[i] -df.index[i-1]).days==0:
                counter=counter+1
            else:
                counter=0  
            if counter>=sequencelength-1: 
                dfnew = DataFrame(data=None, columns=df.columns, index=None)
                dfnew=dfnew.append(df.iloc[i-sequencelength+1:i+1,])
                #now reshape each sample of this sequence
                dfnew=configure_steps(dfnew,varIndex,delays)
                dfvalue=dfnew.values
                X, y = dfvalue[:, 0:dfvalue.shape[1]-1], dfvalue[:,  -1]
                inputSeq = X.reshape(1,(sequencelength-1)*(X.shape[1]))
                outputSeq=y[-1]
                if inputArray is None:
                    inputArray=np.array(inputSeq)
                    outputArray=np.array(outputSeq)
                else:
                    inputArray=np.concatenate((inputArray,inputSeq))
                    outputArray=np.hstack((outputArray,outputSeq))
#                restart the counter to wait for next sequence
                counter=0
    return inputArray,outputArray

def plot_variables(df):
    values=df.values
    groups = np.arange(0, values.shape[1], 1) 
    i = 1
    # plot each column
    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(df.columns[group], y=0.5, loc='right')
        i += 1
    pyplot.show()
    return

def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save("LSTM_model.h5")

def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    return loaded_model

def save_sklearnmodel(model):
    # save the model to disk
    filename = 'model.sav'
    pickle.dump(model, open(filename, 'wb'))
    
def load_sklearnmodel():
    # save the model to disk
    filename = 'model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

def subsequencing(df):
    counter = 0
    dflist = []
    for i in range(len(df.index)):
        if df.index[i] + DateOffset(minutes=30) not in df.index:
            dflist.append(df.iloc[counter:(i + 1), :])
            counter = i + 1
    return dflist

def inputreshaper(X, time_steps=6,N=6):
    totalArray = []
    m, n = X.shape

    for i in range(time_steps):
        # copying
        temp = np.copy(X)
        # shifting
        temp = temp[i:m + i - time_steps + 1, :]
        # appending
        totalArray.append(temp)

    # reshaping collated array to (samples, time_steps, dimensions)
    collatedArray = np.concatenate(totalArray, axis=1)
    X_reshaped = collatedArray.reshape((collatedArray.shape[0], time_steps, n))
    X_reshaped = X_reshaped[0:1-N,:,:]#We are removing last N-1 data due to predicting sequence of length N

    return X_reshaped

def outputreshaper(y,N=6):
    totalArray = []
    m = y.shape[0]#since y is (m, )

    for i in range(N):
        # copying
        temp = np.copy(y)
        # shifting
        temp = temp[i : m + i - N + 1]
        # appending
        totalArray.append(temp.reshape(-1,1))#reshaping needed for concatenating along axis=1

    # reshaping collated array to (samples, time_steps)
    collatedArray = np.concatenate(totalArray, axis=1)
    y_reshaped = collatedArray.reshape((collatedArray.shape[0], N))

    return y_reshaped





