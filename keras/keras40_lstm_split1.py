import numpy as np
from keras.models import Sequential, Model #keras의 씨퀀셜 모델로 하겠다
from keras.layers import Dense, LSTM, Input # Dense와 LSTM 레이어를 쓰겠다

# 멀 자르라는 건데!?
# 비는 시간에 그 책에서 선형대수쪽 공부하면 될 듯 싶어
# 모델을 항상 가장 간단하게 -> 간단해야 응용을 할 수 있다
# 덕지덕지 붙은건 초기 모델이다
a = np.array(range(1,11))
size = 5
def split_x(seq, size) :
    aaa = []
    for i in range(len(seq) - size + 1) :
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa)) 
    return np.array(aaa)
dataset = split_x(a, size)

x = []
y = []
for i in dataset :
    y.append(i[-1])
for i in dataset :
    x.append(list(i[:4]))
print(x) 
print(y)

x = np.array(x)
y = np.array(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.23, shuffle=False
)

#2. 모델구성
from keras.models import Sequential 
from keras.layers import Dense 
model = Sequential()
model.add(Dense(5, input_dim=4)) 
model.add(Dense(16)) 
model.add(Dense(32))  
model.add(Dense(1)) 

#3. 알아듣게 설명한 후 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train, epochs=500, batch_size=3)  

#4. 평가와 예측
loss,mse = model.evaluate(x_test,y_test) 
print('mse 는',mse)

# 예측값 y햇 만들어주기
y_predict = model.predict(x_test)
print(y_predict)

from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
    
print('MSE 는', mean_squared_error(y_test, y_predict) )
print('RMSE 는', RMSE(y_test, y_predict) )

# R2 구하기
from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)
print('R2는 ', r2)
