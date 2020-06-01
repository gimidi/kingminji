'''
(모델 밑,컴파일 위)에서 model.save 했을때
-> 에포 그대로 돌아감 -> 즉 결과값 다르겠지
loss 는 0.05999820324950852
acc 는 0.9811000227928162
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

# 모델 불러오기
from keras.models import load_model
model = load_model('./model/model_test01.h5')
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 
model.fit(x_train,y_train, epochs=10, batch_size=500, validation_split=0.2)

# 평가 및 예측
loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)





