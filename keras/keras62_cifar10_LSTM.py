<<<<<<< HEAD
'''
acc = 21
'''
from keras.datasets import cifar10, mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Dropout, MaxPool2D, Flatten
import matplotlib.pyplot as plt

(x_train, y_train),(x_test,y_test) = cifar10.load_data()

# 전처리 (min_max스케일링, 원-핫 인코딩)
x_train = (x_train/255)
x_test = (x_test/255)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

# LSTM
x_train2 = x_train.reshape(50000, 32*32, 3)
x_test2 = x_test.reshape(10000, 32*32, 3)
model = Sequential()
model.add(LSTM(8, input_shape=(32*32,3))) 
model.add(Dense(16))
model.add(Dropout(0.2)) 
model.add(Dense(64))
model.add(Dense(82))
model.add(Dropout(0.2)) 
model.add(Dense(1260))
model.add(Dense(32))
model.add(Dropout(0.6))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 
model.fit(x_train2,y_train, epochs=10, batch_size=1000, verbose=1)
loss, acc = model.evaluate(x_test2,y_test) 
print('loss 는',loss)
=======
'''
acc = 21
'''
from keras.datasets import cifar10, mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Dropout, MaxPool2D, Flatten
import matplotlib.pyplot as plt

(x_train, y_train),(x_test,y_test) = cifar10.load_data()

# 전처리 (min_max스케일링, 원-핫 인코딩)
x_train = (x_train/255)
x_test = (x_test/255)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

# LSTM
x_train2 = x_train.reshape(50000, 32*32, 3)
x_test2 = x_test.reshape(10000, 32*32, 3)
model = Sequential()
model.add(LSTM(8, input_shape=(32*32,3))) 
model.add(Dense(16))
model.add(Dropout(0.2)) 
model.add(Dense(64))
model.add(Dense(82))
model.add(Dropout(0.2)) 
model.add(Dense(1260))
model.add(Dense(32))
model.add(Dropout(0.6))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 
model.fit(x_train2,y_train, epochs=10, batch_size=1000, verbose=1)
loss, acc = model.evaluate(x_test2,y_test) 
print('loss 는',loss)
>>>>>>> fa2732da2f961158aee21e0eecf6dd0dd4f77931
print('acc 는',acc)