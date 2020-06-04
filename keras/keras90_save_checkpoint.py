<<<<<<< HEAD
'''
loss 는 0.06756242459287168
acc 는 0.9783999919891357
'''
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
x_train = (x_train/255).reshape(60000, 28, 28, 1)
x_test = (x_test/255).reshape(10000, 28, 28, 1)

# 모델구성
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Conv2D(16, (2,2), input_shape=(28,28,1), padding='same')) 
model.add(Conv2D(32, (2,2), padding='same')) 
model.add(MaxPooling2D(2))
model.add(Dropout(0.2))
model.add(Conv2D(43, (2,2), padding='same')) 
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

# 훈련
early_stop = EarlyStopping(monitor='loss', patience=100, mode='auto')
modelpath = './model/{epoch:03d}-{acc:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_acc', verbose=1  ############################## 주인공이심
                            ,save_best_only=True, save_weights_only=False)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 
model.fit(x_train,y_train, epochs=20, batch_size=500, validation_split=0.2
                , callbacks = [early_stop, checkpoint]) 

# 평가 및 예측
loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)
=======
'''
loss 는 0.06756242459287168
acc 는 0.9783999919891357
'''
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
x_train = (x_train/255).reshape(60000, 28, 28, 1)
x_test = (x_test/255).reshape(10000, 28, 28, 1)

# 모델구성
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Conv2D(16, (2,2), input_shape=(28,28,1), padding='same')) 
model.add(Conv2D(32, (2,2), padding='same')) 
model.add(MaxPooling2D(2))
model.add(Dropout(0.2))
model.add(Conv2D(43, (2,2), padding='same')) 
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

# 훈련
early_stop = EarlyStopping(monitor='loss', patience=100, mode='auto')
modelpath = './model/{epoch:03d}-{acc:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_acc', verbose=1  ############################## 주인공이심
                            ,save_best_only=True, save_weights_only=False)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 
model.fit(x_train,y_train, epochs=20, batch_size=500, validation_split=0.2
                , callbacks = [early_stop, checkpoint]) 

# 평가 및 예측
loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)
>>>>>>> fa2732da2f961158aee21e0eecf6dd0dd4f77931
