'''
loss 는 0.06251375317443163
acc 는 0.980400025844574
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

'''
여기에 model 이랑 웨이트 저장 같이되는거면 대박이네.......!?
와....대박사건인데?......
'''
from keras.models import load_model
model = load_model('./model/004-0.9653.hdf5')
# 이상태에서 훈련을 한다는것은 저 hdf5에서 model, compile, fit 옵션을 가지고 있다는거임
# load_model 또는 weight가 필요가 없겠는데? -> 아니래!
# 왜냐면 얘 체크포인트는 가장 높은 값까지 알려주잖아

# 평가 및 예측
loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)
