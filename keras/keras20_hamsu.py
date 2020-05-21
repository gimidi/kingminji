'''
MSE 는 3.725290298461914e-09
RMSE 는 6.103515625e-05
R2는  0.9999999999153343
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
from keras.models import Sequential, Model 
from keras.layers import Dense, Input 

# Sequential 모델
# model = Sequential()
# model.add(Dense(5, input_dim=3)) 
# model.add(Dense(4)) 
# model.add(Dense(1)) 

# 함수형 모델
input1 = Input(shape=(3, ))
dense1 = Dense(5, activation='relu')(input1) # input 레이어를 알려줘야 함/ 노드를 5개로 가지고 input1을 인풋레이어로 받는 댄스층
dense2 = Dense(8, activation='relu')(dense1)
dense3 = Dense(32, activation='relu')(dense2)
dense4 = Dense(8, activation='relu')(dense3)
output1 = Dense(1)(dense4)

model = Model(inputs=input1, outputs=output1)  # 함수형 모델임을 명시 레알 개신기하네...

model.summary()


#3. 알아듣게 설명한 후 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train, epochs=3000, batch_size=5,
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
