'''
mse는 5028.216796875
RMSE 는 70.90992225446944
R2는  0.22524102084842024
'''
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x = dataset.data
y = dataset.target

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(); scaler.fit(x); x = scaler.transform(x) 

from sklearn.decomposition import PCA
pca = PCA(n_components=8); pca.fit(x); x = pca.transform(x)

# test 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.2)

import numpy as np
from keras.models import Sequential, Model 
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LSTM

# lstm
x_train = x_train.reshape(-1,4,2)
x_test = x_test.reshape(-1,4,2)

model = Sequential()
model.add(LSTM(1600, input_shape=(4,2)))
model.add(Dense(1600))
model.add(Dropout(0.2))

model.add(Dense(640))
model.add(Dense(640))
model.add(Dropout(0.4))

model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
hist = model.fit(x_train, y_train, epochs=50, batch_size=8, verbose=2, validation_split=0.2)
loss, mse = model.evaluate(x_test, y_test)
print('mse는', mse)

# RMSE , R2
from sklearn.metrics import mean_squared_error
y_predict = model.predict(x_test)
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE 는', RMSE(y_test, y_predict ))

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