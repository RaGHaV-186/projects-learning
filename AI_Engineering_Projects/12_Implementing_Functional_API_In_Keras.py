import tensorflow as tf
from keras.src.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,BatchNormalization

input_layer = Input(shape=(20,))

hidden_layer1 = Dense(64,activation='relu')(input_layer)
hidden_layer2 = Dense(64, activation='relu')(hidden_layer1)

output_layer = Dense(1,activation='sigmoid')(hidden_layer2)

model = Model(inputs= input_layer,outputs=output_layer)

print(model.summary())

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Dropout Layers and Batch Normalization

input_layer = Input(shape=(20,))

hidden_layer = Dense(64,activation='relu')(input_layer)

dropout_layer = Dropout(rate=0.5)(hidden_layer)

hidden_layer2 = Dense(64,activation='relu')(dropout_layer)

output_layer = Dense(1,activation='sigmoid')(hidden_layer2)

model = Model(inputs =input_layer,outputs=output_layer)

print(model.summary())

input_layer = Input(shape=(20,))

hidden_layer = Dense(64, activation='relu')(input_layer)

batch_norm_layer = BatchNormalization()(hidden_layer)

hidden_layer2 = Dense(64, activation='relu')(batch_norm_layer)

output_layer = Dense(1, activation='sigmoid')(hidden_layer2)

model = Model(inputs=input_layer, outputs=output_layer)

print(model.summary())