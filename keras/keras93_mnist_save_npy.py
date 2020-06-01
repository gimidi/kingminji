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

np.save('./data/mnist__x_train.npy', arr=x_train)
np.save('./data/mnist__y_train.npy', arr=y_train)
np.save('./data/mnist__x_test.npy', arr=x_test)
np.save('./data/mnist__y_test.npy', arr=y_test)

