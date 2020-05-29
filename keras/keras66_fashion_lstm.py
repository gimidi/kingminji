from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Dropout, MaxPool2D, Flatten
import matplotlib.pyplot as plt

(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data() #6만/ 1만개/ 3차원

# 전처리
x_train = (x_train/255)
x_test = (x_test/255)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# LSTM
model = Sequential()
model.add(LSTM(8, input_shape=(28,28))) 
model.add(Dense(16))
model.add(Dropout(0.2)) 
model.add(Dense(64))
model.add(Dense(82))
model.add(Dropout(0.2)) 
model.add(Dense(126))
model.add(Dense(32))
model.add(Dropout(0.6))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 
model.fit(x_train,y_train, epochs=100, batch_size=1000, verbose=1)
loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)

'''
loss 는 0.4895001473426819
acc 는 0.8299999833106995
'''