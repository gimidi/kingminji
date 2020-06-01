from sklearn.datasets import load_iris
import numpy as np
datasets = load_iris()

print(type(datasets.data))  # <class 'numpy.ndarray'>

x_data = datasets.data
y_data = datasets.target

np.save('./data/iris_x.npy', arr=x_data)
np.save('./data/iris_y.npy', arr=y_data)

x_data_load = np.load('./data/iris_x.npy') # 굳이 저장해서 쓰는 이유는 뭐지?
y_data_load = np.load('./data/iris_y.npy')

print(x_data_load)
print(x_data_load.shape)
print(y_data_load)
print(y_data_load.shape)


