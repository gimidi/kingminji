# 10가지 이미지 중 정답 찾는거고 -> y 원-핫 인코딩
# 칼라임 -> 채널은 3  /  (50000, 32, 32, 3) , (10000, 32, 32, 3)

from keras.datasets import cifar10, mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Dropout, MaxPool2D, Flatten
import matplotlib.pyplot as plt

(x_train, y_train),(x_test,y_test) = cifar10.load_data()

print(x_train[0])
print(y_train[0])

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

plt.imshow(x_train[0])
plt.show()

