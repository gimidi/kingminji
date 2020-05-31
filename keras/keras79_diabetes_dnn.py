'''
mse는 3580.053955078125
RMSE 는 59.83355399565519
R2는  0.44837721128269625
'''
from sklearn.datasets import load_diabetes

dataset = load_diabetes()

x = dataset.data
y = dataset.target

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x) 

from sklearn.decomposition import PCA
pca = PCA(n_components=8)
pca.fit(x)
x = pca.transform(x)

# test 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.2)

import numpy as np
from keras.models import Sequential, Model 
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LSTM

# dnn
model = Sequential()
model.add(Dense(1600, input_shape=(6,)))
model.add(Dense(1600))
model.add(Dropout(0.2))

model.add(Dense(640))
model.add(Dense(640))
model.add(Dropout(0.5))

model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
hist = model.fit(x_train, y_train, epochs=100, batch_size=4, verbose=2, validation_split=0.2)

loss, mse = model.evaluate(x_test, y_test)
print('mse는', mse)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
y_predict = model.predict(x_test)
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE 는', RMSE(y_test, y_predict ))

# R2 구하기
from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)
print('R2는 ', r2)

# 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(12,10))
# plt.subplot(2,1,1) # 2행 1열 에 첫번째꺼 그리겠다
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='gray', label='val_loss')
plt.grid()
plt.xlabel('epoch', size=15)
plt.ylabel('loss', size=15)
plt.legend(loc='upper left')
plt.show()