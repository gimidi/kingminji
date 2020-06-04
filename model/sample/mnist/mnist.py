'''
loss 는 0.027926341804021537
acc 는 0.9940000176429749
'''
import pandas as pd
import numpy as np

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 전처리 시작
x_train = (x_train/255).reshape(-1, 28, 28, 1)
x_test = (x_test/255).reshape(-1, 28, 28, 1)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 모델구성
from keras.models import Sequential, save_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(28,28,1), padding='same', activation='relu'))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2)) 
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
model.add(Conv2D(128, (2,2), padding='same', activation='relu')) 
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()
model.save('./model/sample/mnist/model.h5') 

from keras.callbacks import EarlyStopping, ModelCheckpoint

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])  # 성능 훨씬 좋음 -> 사실상 binary를 사용하는게 맞다는거임
early_stop = EarlyStopping(monitor='acc', patience=100, mode='auto')
modelpath = './model/sample/mnist/{epoch:02d}-{acc:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1  
                            ,save_best_only=True, save_weights_only=False)
model.fit(x_train,y_train, epochs=50, batch_size=900, validation_split=0.1, verbose=2
            ,callbacks = [early_stop, checkpoint])  

model.save_weights('./model/sample/mnist/model_weight.h5')

# 평가와 예측
loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)
