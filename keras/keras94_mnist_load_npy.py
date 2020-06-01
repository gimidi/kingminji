'''
loss 는 
acc 는 
90번 save_checkpoint와 같음
'''
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

x_train = np.load('./data/mnist__x_train.npy')
y_train = np.load('./data/mnist__y_train.npy')
x_test = np.load('./data/mnist__x_test.npy')
y_test = np.load('./data/mnist__y_test.npy')


from keras.models import load_model
model = load_model('./model/14-0.0690.hdf5')

loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)
