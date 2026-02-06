import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)
import numpy as np
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Input

cali = fetch_california_housing()

cali_data = cali.data

target = cali.target

print(cali_data)

print(target)

col_names = cali.feature_names

print(col_names)

df = pd.DataFrame(cali_data,columns=col_names)

df['Price'] = target

print(df.head())

X = df.drop('Price',axis=1)

y = df['Price']

print(X.shape)

print(df.describe())

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print(X_test.shape)
print(X_train.shape)

input_shape = X_train.shape[1]

def build_model():

    model = Sequential()

    model.add(Input(shape=(input_shape,)))

    model.add(Dense(30,activation='relu'))
    model.add(Dense(30,activation='relu'))

    model.add(Dense(1))

    model.compile(optimizer='adam',loss='mean_squared_error')

    return model


model = build_model()

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, verbose=2)
