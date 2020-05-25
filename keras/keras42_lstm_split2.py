'''
걍 lstm 쓰레기임
100개로 돌려도 쓰레기임..!!!

mse 는 115.99845886230469
[[82.097115]
 [82.36568 ]
 [82.61987 ]
 [82.86034 ]
 [83.08766 ]
 [83.30248 ]]
RMSE 는 14.834567084606148
R2는  -74.45064477291889

데이터 차원을 3개로 늘린게 잘 안맞는거 같아 
뭔가 이런 숫자 데이터에 잘 안맞는거 아닐까...??
'''

import numpy as np

a = np.array(range(1,101))
size = 5
def split_x(seq, size) :
    aaa = []
    for i in range(len(seq) - size + 1) :
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa)) 
    return np.array(aaa)

dataset = split_x(a, size)
x = np.array(dataset[:,:4])
y = np.array(dataset[:,4])
x_predict = x[-6:,:]
x_predict_y = y[-6:]

x = x.reshape(x.shape[0],x.shape[1],1)
x_predict = x_predict.reshape(x_predict.shape[0],x_predict.shape[1],1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.2, shuffle=False
)

#2. 모델구성
from keras.models import Sequential 
from keras.layers import LSTM, Dense 

model = Sequential()
model.add(LSTM(320, input_shape=(4,1))) # input을 넣는거야 무조권 ^_^ 
model.add(Dense(32))
model.add(Dense(640))
model.add(Dense(320))
model.add(Dense(16))  
model.add(Dense(1)) 

#3. 설명한 후 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=500, mode='auto') # 어느정도 사이즈부터 성능향상에 도움을 주는지 아직 머르겠씁니다
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train, epochs=500, batch_size=5, callbacks=[early_stopping], validation_split=0.2)  

#4. 평가와 예측
loss,mse = model.evaluate(x_test,y_test) 
print('mse 는',mse)

x_predict = model.predict(x_predict)
print(x_predict)

from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE 는', RMSE(x_predict_y, x_predict) )

# R2 구하기
from sklearn.metrics import r2_score 
r2 = r2_score(x_predict_y, x_predict)
print('R2는 ', r2)
