#1. 데이터
import numpy as np
x = np.array([range(1,101), range(311,411), range(100)]).T  # 여러칼럼 넣고싶으면 대괄호 [] , 들어갈 수 있는 요소의 갯수가 2가 최대이다.
y = np.array([range(101,201), range(711,811), range(100)]).T

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.23, shuffle=False
)

#2. 모델구성
from keras.models import Sequential 
from keras.layers import Dense 
model = Sequential()
model.add(Dense(5, input_dim=3)) 
model.add(Dense(8)) 
model.add(Dense(16)) 
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(5))
model.add(Dense(3)) # 와우.. 절대 몰랐네.. 주피터 없으면 어쩔거여..?!/ 주피터로 데이터구조를 보지 않으면 아무것도 할 수가 없어따,, 또르르,,,

'''
# print(x_train)
# # print(x_val)
# print(x_test)

'''

#3. 알아듣게 설명한 후 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train, epochs=600, batch_size=15,
         validation_split = 0.23)
        
#4. 평가와 예측
loss,mse = model.evaluate(x_test,y_test)
print('loss(mse) 는',loss)

y_predict = model.predict(x_test)
print('y_predict 값은', y_predict)

# +추가지표인 RMSE와 R2 구하기
from sklearn.metrics import mean_squared_error # MSE임
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

print('MSE 는', mean_squared_error(y_test, y_predict) )
print('RMSE 는', RMSE(y_test, y_predict) )

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2는 ', r2)
