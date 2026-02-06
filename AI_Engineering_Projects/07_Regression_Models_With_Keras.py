import keras
import pandas as pd
from google.protobuf.descriptor import Descriptor

pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)
import numpy as np
from sklearn.preprocessing  import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input


filepath='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv'
concrete_data = pd.read_csv(filepath)

print(concrete_data.head())

print(concrete_data.shape)

print(concrete_data.describe())

print(concrete_data.isnull().sum())

concrete_data_columns = concrete_data.columns

predictors = concrete_data.drop(columns=['Strength'])
target = concrete_data['Strength']

print(predictors.head())

print(target.head())

scaler = StandardScaler()

predictors_norm = scaler.fit_transform(predictors)

predictors_norm = pd.DataFrame(predictors_norm, columns=predictors.columns)

print(predictors_norm.head())

n_cols = predictors_norm.shape[1]

def regression_model():

    model = Sequential()
    model.add(Input(shape=(n_cols)))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam',loss='mean_squared_error')

    return model

model = regression_model()

model.fit(predictors_norm,target,validation_split=0.3,epochs=100,verbose=2)


