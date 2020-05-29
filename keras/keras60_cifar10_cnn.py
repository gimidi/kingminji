'''
acc = 83 %
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

# CNN
model = Sequential()
model.add(Conv2D(16, (2,2), input_shape=(32,32,3), padding='same'))
model.add(Conv2D(16, kernel_size = (5,5), padding = 'Same',activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3,3), padding='same'))
model.add(Conv2D(32, kernel_size = (5,5), padding = 'same',activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# model.add(Conv2D(32, (2,2), padding='same',activation ='relu'))
# model.add(Conv2D(32, (2,2), padding='same',activation ='relu'))
# model.add(MaxPooling2D(2,2)) 
# model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 
model.fit(x_train,y_train, epochs=350, batch_size=1000, verbose=2)
loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)

