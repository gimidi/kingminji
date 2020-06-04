<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

plt.imshow(x_train[0],'gray')
plt.imshow(x_train[0])
plt.show()

print(x_train[0]) #[0,0,0,0,....................0,0,.....]
print('y_train[0]:',y_train[0]) # 5
print(x_train[0].shape) # (28, 28) 짜리

=======
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

plt.imshow(x_train[0],'gray')
plt.imshow(x_train[0])
plt.show()

print(x_train[0]) #[0,0,0,0,....................0,0,.....]
print('y_train[0]:',y_train[0]) # 5
print(x_train[0].shape) # (28, 28) 짜리

>>>>>>> fa2732da2f961158aee21e0eecf6dd0dd4f77931
