<<<<<<< HEAD
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Dropout, MaxPool2D, Flatten
import matplotlib.pyplot as plt

# 데이터
(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()

print(x_train[0])
print(y_train[0])

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

plt.imshow(x_train[0])
=======
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Dropout, MaxPool2D, Flatten
import matplotlib.pyplot as plt

# 데이터
(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()

print(x_train[0])
print(y_train[0])

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

plt.imshow(x_train[0])
>>>>>>> fa2732da2f961158aee21e0eecf6dd0dd4f77931
plt.show()