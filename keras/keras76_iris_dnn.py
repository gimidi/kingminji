'''
acc는 1.0
'''

from sklearn.datasets import load_iris

dataset = load_iris()

x = dataset.data
y = dataset.target

# 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x) 

from keras.utils import np_utils
y = np_utils.to_categorical(y)

# 모델링
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.2 )

import numpy as np
from keras.models import Sequential, Model 
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model = Sequential()
model.add(Dense(16, input_shape=(4,)))
model.add(Dense(16))
model.add(Dropout(0.2))

model.add(Dense(64))
model.add(Dense(64))
model.add(Dropout(0.2))

model.add(Dense(3, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=50, batch_size=2, verbose=2, validation_split=0.2)

loss, acc = model.evaluate(x_test, y_test)
print('acc는', acc)