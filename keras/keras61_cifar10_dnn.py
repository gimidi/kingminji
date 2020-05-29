'''
acc = 39%
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

# DNN
x_train1 = x_train.reshape(-1, 32*32*3)
x_test1 = x_test.reshape(-1, 32*32*3)
model = Sequential()
model.add(Dense(8, input_dim=(32*32*3))) 
model.add(Dense(16))
model.add(Dense(64))
model.add(Dense(120))
model.add(Dropout(0.4))
model.add(Dense(120))
model.add(Dropout(0.4))
model.add(Dense(32))
model.add(Dense(10, activation='softmax')) 
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 
model.fit(x_train1,y_train, epochs=100, batch_size=300, verbose=2, validation_split=0.3)
loss, acc = model.evaluate(x_test1,y_test) 
print('loss 는',loss)
print('acc 는',acc)