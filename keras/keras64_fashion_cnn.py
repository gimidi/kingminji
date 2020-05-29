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

# CNN
x_train1 = x_train.reshape(-1,28,28,1)
x_test1 = x_test.reshape(-1,28,28,1)
model = Sequential()
model.add(Conv2D(16, (2,2), input_shape=(28,28,1), padding='same'))
model.add(Conv2D(16, kernel_size = (5,5), padding = 'Same',activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3,3), padding='same'))
model.add(Conv2D(32, kernel_size = (5,5), padding = 'same',activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))

# model.add(Conv2D(32, (2,2), padding='same',activation ='relu'))
# model.add(Conv2D(32, (2,2), padding='same',activation ='relu'))
# model.add(MaxPooling2D(2,2)) 
# model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 
model.fit(x_train1,y_train, epochs=350, batch_size=700, verbose=1)
loss, acc = model.evaluate(x_test1,y_test) 
print('loss 는',loss)
print('acc 는',acc)

'''
loss 는 0.24341917735841126
acc 는 0.935699999332428
'''