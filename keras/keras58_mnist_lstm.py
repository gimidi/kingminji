<<<<<<< HEAD
'''
acc = 98% 이상 !
'''
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 전처리
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#버전 1 -> acc 89%
x_train = (x_train/255).reshape(-1, 28*28,1)
x_test = (x_test/255).reshape(-1, 28*28, 1)

# 버전 2 -> acc 90% 속도 훨씬 빠름 / 근데 위에꺼가 데이터상 맞는 구조 아냐?
# x_train = (x_train/255).reshape(-1, 28,28)
# x_test = (x_test/255).reshape(-1, 28, 28)

# 모델구성
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(LSTM(8, input_shape=(28*28,1))) 
model.add(Dense(16))
model.add(Dense(64))
model.add(Dense(82))
model.add(Dense(1260))
model.add(Dense(32))
model.add(Dense(10, activation='softmax')) #와... 뭐지?... 이거 차이를 모르겠네

model.summary()

#3. 설명한 후 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) 
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs=10, batch_size=900)  

#4. 평가와 예측
loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)

predict = model.predict(x_test)
print(np.argmax(predict, axis = 1))





=======
'''
acc = 98% 이상 !
'''
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 전처리
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#버전 1 -> acc 89%
x_train = (x_train/255).reshape(-1, 28*28,1)
x_test = (x_test/255).reshape(-1, 28*28, 1)

# 버전 2 -> acc 90% 속도 훨씬 빠름 / 근데 위에꺼가 데이터상 맞는 구조 아냐?
# x_train = (x_train/255).reshape(-1, 28,28)
# x_test = (x_test/255).reshape(-1, 28, 28)

# 모델구성
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(LSTM(8, input_shape=(28*28,1))) 
model.add(Dense(16))
model.add(Dense(64))
model.add(Dense(82))
model.add(Dense(1260))
model.add(Dense(32))
model.add(Dense(10, activation='softmax')) #와... 뭐지?... 이거 차이를 모르겠네

model.summary()

#3. 설명한 후 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) 
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs=10, batch_size=900)  

#4. 평가와 예측
loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)

predict = model.predict(x_test)
print(np.argmax(predict, axis = 1))





>>>>>>> fa2732da2f961158aee21e0eecf6dd0dd4f77931
