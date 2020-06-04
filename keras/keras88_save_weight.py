<<<<<<< HEAD
'''
(fit 밑)에서 model.save_weights 시켰음
-> fit하고 마지막 가중치 저장 -> 따라서 load로 쓰일때 fit 대용치로 쓰임 -> 돌려서 결과 잘나오면 그 후에 save 시켜서 다시 재현할 수 있음
오히려 가장 의외였던게 같은 가중치에서 무조건 같은 예측치값이 나온다는 것이었음...!
loss 는 0.16832394029647113
acc 는 0.9531000256538391
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
model.fit(x_train,y_train, epochs=5, batch_size=500, validation_split=0.2)

model.save_weights('./model/model_weight01.h5')

# 평가 및 예측
loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)

=======
'''
(fit 밑)에서 model.save_weights 시켰음
-> fit하고 마지막 가중치 저장 -> 따라서 load로 쓰일때 fit 대용치로 쓰임 -> 돌려서 결과 잘나오면 그 후에 save 시켜서 다시 재현할 수 있음
오히려 가장 의외였던게 같은 가중치에서 무조건 같은 예측치값이 나온다는 것이었음...!
loss 는 0.16832394029647113
acc 는 0.9531000256538391
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
model.fit(x_train,y_train, epochs=5, batch_size=500, validation_split=0.2)

model.save_weights('./model/model_weight01.h5')

# 평가 및 예측
loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)

>>>>>>> fa2732da2f961158aee21e0eecf6dd0dd4f77931
