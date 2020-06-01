'''
(fit 밑)에서 model.save 했을때
-> lib import 안함 & 에포 안돌아감 -> 즉 결과값 똑같e
loss 는 0.06347653883234598
acc 는 0.9800000190734863
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
# fit까지 담긴 모델 다음에 danse를 추가하는게 가능해???

# 평가 및 예측
loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)
