#1. 데이터
import numpy as np
x = np.array(range(1,101))
y = np.array(range(101,201))

# 싸이킷런..! 또 너구나..!!
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    # x, y, random_state=66,train_size=0.6
    x, y, random_state=66,shuffle=False, train_size=0.6   
)

x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, random_state=66, test_size=0.5, shuffle=False 
)

# x_train = x[:60] # 6 : 2 : 2 로 분리 -> 개인적인 생각은 test셋으로 버리는 비율을 최소화하는 방향으로 가야되지 않나 싶다.
# x_val = x[60:80]
# x_test = x[80:]

# y_train = y[:60]
# y_val = y[60:80]
# y_test = y[80:]   # 아.. 당연히 이건 index번호니까 하나씩 밀어서 생각해야되네 ㅎㅎ..호호 이걸 헷갈리는구나

# print(x_train)
# print(x_val)
# print(x_test)

#2. 모델구성
from keras.models import Sequential 
from keras.layers import Dense # DNN 구조의 가장 베이스가 되는 모델
model = Sequential()
model.add(Dense(5, input_dim=1)) #인풋레이어/ 5개의 노드 아웃풋/ # 10개짜리 **한 덩어리** input하겠다
model.add(Dense(8)) # 인풋이 5, 아웃풋이 3개 노드인 아웃풋 히든레이어
model.add(Dense(16))  # 줄마다 W가 점점 바뀌고 있는것임
model.add(Dense(32))
model.add(Dense(5))
model.add(Dense(1)) # 아웃레이어

#3. 알아듣게 설명한 후 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train, epochs=250, batch_size=5,
         validation_data=(x_val,y_val)) # 이게 다야?ㅋㅋㅋㅋ 대박이네 

#4. 평가와 예측
loss,mse = model.evaluate(x_test,y_test)  # evaluate반환하는건 loss, metrics 순임!
print('loss(mse) 는',mse)

y_predict = model.predict(x_test)
print('y_predict 값은', y_predict) # 예측값


# RMSE 구하기
from sklearn.metrics import mean_squared_error #MSE임

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('MSE 는', mean_squared_error(y_test, y_predict) )
print('RMSE 는', RMSE(y_test, y_predict) )

# R2 구하기
from sklearn.metrics import r2_score # 땡겼다!
r2 = r2_score(y_test, y_predict)
print('R2는 ', r2)
