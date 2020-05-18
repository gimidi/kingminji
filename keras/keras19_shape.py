'''
열우선 행무시 shape은 3열이고 행은 알바 아니다
shape=(3,)
'''
#1. 데이터
import numpy as np
x = np.array([range(1,101), range(311,411), range(100)]).T # 여러칼럼 넣고싶으면 대괄호 []
y = np.array(range(711,811)).T

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.23 , shuffle=False
)

#2. 모델구성
from keras.models import Sequential 
from keras.layers import Dense 
model = Sequential()
model.add(Dense(5, input_shape=(3,)))
model.add(Dense(1)) 

#3. 알아듣게 설명한 후 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train, epochs=500, batch_size=15,
         validation_split = 0.23, verbose=1) # verbose 학습중의 정보를 보여주는 버전 고르기 0(아예안나옴),1(다나옴),2(화살표는 안나옴),3(epo만 나옴) 딜레이 시간 아낄려고 -> 1/2 쓰면 될듯
        
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
