<<<<<<< HEAD
'''
epoch 가 돈다는건 새로 학습을 한다는건데...
loss 는 0.16832394029647113 -> 0.16832394029647113
acc 는 0.9531000256538391 -> 0.9531000256538391
왜 결과가 똑같은거야....??
그럼 fit 할 필요가 없다는거네
근데 너무 이상하다.. 가중치는 저장을 했는디 모델이랑 컴파일은 필요하다?
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
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Dropout,MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Conv2D(16, (2,2), input_shape=(28,28,1), padding='same')) 
model.add(Conv2D(16, (2,2), padding='same')) 
model.add(MaxPooling2D(2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

# 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 
 
# model.fit(x_train,y_train, epochs=5, batch_size=500, validation_split=0.2)

model.load_weights('./model/model_weight01.h5')

# 평가 및 예측
loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)

=======
'''
epoch 가 돈다는건 새로 학습을 한다는건데...
loss 는 0.16832394029647113 -> 0.16832394029647113
acc 는 0.9531000256538391 -> 0.9531000256538391
왜 결과가 똑같은거야....??
그럼 fit 할 필요가 없다는거네
근데 너무 이상하다.. 가중치는 저장을 했는디 모델이랑 컴파일은 필요하다?
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
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Dropout,MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Conv2D(16, (2,2), input_shape=(28,28,1), padding='same')) 
model.add(Conv2D(16, (2,2), padding='same')) 
model.add(MaxPooling2D(2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

# 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 
 
# model.fit(x_train,y_train, epochs=5, batch_size=500, validation_split=0.2)

model.load_weights('./model/model_weight01.h5')

# 평가 및 예측
loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)

>>>>>>> fa2732da2f961158aee21e0eecf6dd0dd4f77931
