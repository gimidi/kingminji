<<<<<<< HEAD
from sklearn.datasets import load_boston
# data : x값
# target : y값

dataset = load_boston()
x = dataset.data
y = dataset.target

# 전처리
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
    x, y, random_state=66, test_size=0.23 , shuffle=False)

import numpy as np
from keras.models import Sequential, Model 
from keras.layers import Dense, LSTM, Input, Dropout, Conv2D, MaxPooling2D

# dnn
input1 = Input(shape=(8,))
layer1 = Dense(320)(input1)
layer2 = Dense(640)(layer1)
layer3 = Dropout(0.2)(layer2)

layer4 = Dense(1280)(layer2)
layer5 = Dense(128)(layer4)
layer6 = Dropout(0.2)(layer5)
layer5 = Dense(32)(layer4)

output1 = Dense(1)(layer5)

model = Model(inputs=input1, outputs=output1)
model.summary()

# 설명과 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
hist = model.fit(x_train,y_train, epochs=50, batch_size=5, validation_split=0.2)
                #,callbacks=[early_stop, checkpoint])

loss, mse = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('mse 는',mse)

# +추가지표인 RMSE와 R2 구하기
y_predict = model.predict(x_test)
from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

print('RMSE 는', RMSE(y_test, y_predict) )

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
=======
from sklearn.datasets import load_boston
# data : x값
# target : y값

dataset = load_boston()
x = dataset.data
y = dataset.target

# 전처리
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
    x, y, random_state=66, test_size=0.23 , shuffle=False)

import numpy as np
from keras.models import Sequential, Model 
from keras.layers import Dense, LSTM, Input, Dropout, Conv2D, MaxPooling2D

# dnn
input1 = Input(shape=(8,))
layer1 = Dense(320)(input1)
layer2 = Dense(640)(layer1)
layer3 = Dropout(0.2)(layer2)

layer4 = Dense(1280)(layer2)
layer5 = Dense(128)(layer4)
layer6 = Dropout(0.2)(layer5)
layer5 = Dense(32)(layer4)

output1 = Dense(1)(layer5)

model = Model(inputs=input1, outputs=output1)
model.summary()

# 설명과 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
hist = model.fit(x_train,y_train, epochs=50, batch_size=5, validation_split=0.2)
                #,callbacks=[early_stop, checkpoint])

loss, mse = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('mse 는',mse)

# +추가지표인 RMSE와 R2 구하기
y_predict = model.predict(x_test)
from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

print('RMSE 는', RMSE(y_test, y_predict) )

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
>>>>>>> fa2732da2f961158aee21e0eecf6dd0dd4f77931
print('R2는 ', r2)