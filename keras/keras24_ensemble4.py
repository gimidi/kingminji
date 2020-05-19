'''
keras21_ensemble3
모델 1의 MSE 는 17.728510218206793
모델 2의 MSE 는 3.6834538328519555
모델1의 RMSE 는 4.210523746306009
모델2의 RMSE 는 1.919232615618012
모델1의 R2는  0.46681172276069793
모델2의 R2는  0.8892194335984374
'''

#1. 데이터
import numpy as np
x1 = np.array([range(1,101), range(301,401)]).T

y1 = np.array([range(711,811), range(611, 711)]).T
y2 = np.array([range(101,201), range(411, 511)]).T

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, y1, y2, random_state=66, test_size=0.2, shuffle=False
)

#2. 모델구성
from keras.models import Sequential, Model 
from keras.layers import Dense, Input 
# 함수형 모델 1
input1 = Input(shape=(2, ))
dense1_1 = Dense(8, activation='relu')(input1)
dense1_2 = Dense(16, activation='relu')(dense1_1)
dense1_3 = Dense(32, activation='relu')(dense1_2)
dense1_4 = Dense(16, activation='relu')(dense1_3) 


# 엮어주쟈~~~ 
# from keras.layers.merge import concatenate
# merge1 = concatenate([dense1_3,dense2_3]) #output 자체를 묶어준다
# middle1 = Dense(32)(merge1)
# middle2 = Dense(64)(middle1)

# output 모델 구성
# 함수형 모델 1
output1_1 = Dense(80)(dense1_4)
output1_2 = Dense(64)(output1_1)
output1_3 = Dense(32)(output1_2)
output1_4 = Dense(2)(output1_3)

# 함수형 모델 2
output2_1 = Dense(80)(dense1_4)
output2_2 = Dense(64)(output2_1)
output2_3 = Dense(32)(output2_2)
output2_4 = Dense(2)(output2_3)

model = Model(inputs=[input1], outputs=[output1_4,output2_4])
model.summary()

#3. 알아듣게 설명한 후 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x1_train,[y1_train,y2_train], epochs=250, batch_size=5,verbose=2)
         #,validation_split = 0.25)
        
#4. 평가
evaluate = model.evaluate(x1_test,[y1_test,y2_test], batch_size=5)
print('evaluate 는',evaluate)
print('전체 loss 는',evaluate[0])
print('모델1의 loss 는',evaluate[1])
print('모델1의 metrics 는',evaluate[2])
print('모델2의 loss 는',evaluate[3])
print('모델2의 metrics 는',evaluate[4])


#5. 예측
y_predict = model.predict(x1_test)
print('y_predict 값은', y_predict)

# +추가지표인 MSE와 RMSE 구하기
from sklearn.metrics import mean_squared_error # MSE임
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

print('모델 1의 MSE 는', mean_squared_error(y1_test, y_predict[0]) )
print('모델 2의 MSE 는', mean_squared_error(y2_test, y_predict[1]) )

print('모델1의 RMSE 는', RMSE(y1_test, y_predict[0]) )
print('모델2의 RMSE 는', RMSE(y2_test, y_predict[1]) )

# R2 구하기
from sklearn.metrics import r2_score
print('모델1의 R2는 ', r2_score(y1_test, y_predict[0]))
print('모델2의 R2는 ', r2_score(y2_test, y_predict[1]))
