from sklearn.datasets import load_boston
# data : x값
# target : y값

dataset = load_boston()
x = dataset.data
y = dataset.target

# x값에 로버스트를 좀 해보자
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x) #  전처리를 실행하다 
x = scaler.transform(x) # 실행한 값을 변환해라

from sklearn.decomposition import PCA
pca = PCA(n_components=8)
pca.fit(x)
x = pca.transform(x) 

# test랑 분리해주기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.23 , shuffle=False
)

import numpy as np
from keras.models import Sequential, Model 
from keras.layers import Dense, LSTM, Input, Dropout

# LSTM
x_train = x_train.reshape(-1,8,1)
x_test = x_test.reshape(-1,8,1)

model = Sequential()
model.add(LSTM(8, input_shape=(8,1))) 
model.add(Dense(16))
model.add(Dropout(0.2)) 
model.add(Dense(64))
model.add(Dense(82))
model.add(Dropout(0.2)) 
model.add(Dense(126))
model.add(Dense(32))
model.add(Dropout(0.6))
model.add(Dense(1, activation='softmax'))
model.summary()

# 설명과 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse','acc'])
hist = model.fit(x_train,y_train, epochs=300, batch_size=2, validation_split=0.2)
                #,callbacks=[early_stop, checkpoint])

loss, mse = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('mse 는',mse)

# +추가지표인 RMSE와 R2 구하기
y_predict = model.predict(x_test)
from sklearn.metrics import mean_squared_error # MSE임
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

print('MSE 는', mean_squared_error(y_test, y_predict) )
print('RMSE 는', RMSE(y_test, y_predict) )

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2는 ', r2)