'''
MSE 는 7.4040144681930535e-09
RMSE 는 8.604658312909964e-05
R2는  0.9999999997773228
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
fit할때 features 는 callbacks=[early_stopping]
'''

#1. 데이터
import numpy as np
x1 = np.array([range(1,101), range(311,411), range(411,511)]).T 
x2 = np.array([range(711,811), range(811,911),range(511,611)]).T
y1 = np.array([range(101,201), range(411,511)]).T 

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(
    x1, x2, y1, random_state=66, test_size=0.2, shuffle=False
)

#2. 모델구성
from keras.models import Sequential, Model 
from keras.layers import Dense, Input 
# 함수형 모델 1
input1 = Input(shape=(3, ))
dense1_1 = Dense(5, activation='relu')(input1) 
dense1_2 = Dense(8, activation='relu')(dense1_1)
dense1_3 = Dense(16, activation='relu')(dense1_2) 

# 함수형 모델 2
input2 = Input(shape=(3, ))
dense2_1 = Dense(5, activation='relu')(input2) 
dense2_2 = Dense(8, activation='relu')(dense2_1)
dense2_3 = Dense(16, activation='relu')(dense2_2)

# 엮어주쟈~~~ 
from keras.layers.merge import concatenate
merge1 = concatenate([dense1_3,dense2_3]) #output 자체를 묶어준다
middle1 = Dense(32)(merge1)
middle2 = Dense(64)(middle1)

# output 모델 구성
# 함수형 모델 1
output1 = Dense(64)(middle2)
output1_2 = Dense(32)(output1)
output1_3 = Dense(2)(output1_2)

# # 함수형 모델 2
# output2 = Dense(16, name='aaaa1')(middle3)
# output2_2 = Dense(8, name='bbbb1')(output2)
# output2_3 = Dense(2, name='cccc1')(output2_2)

model = Model(inputs=[input1, input2], outputs=[output1_3])
model.summary()

#3. 알아듣게 설명한 후 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping #자바 좀 했나보네?
early_stopping = EarlyStopping(monitor='loss', patience=25, mode='auto')
model.fit([x1_train,x2_train],y1_train, epochs=1200, batch_size=5,validation_split = 0.25
        ,callbacks=[early_stopping])
        
#4. 평가
evaluate = model.evaluate([x1_test,x2_test],y1_test, batch_size=5)
print('loss 는',evaluate[0])
print('metrics 는',evaluate[1])


#5. 예측
y_predict = model.predict([x1_test,x2_test])
print('y_predict 값은', y_predict)

# +추가지표인 MSE와 RMSE 구하기
from sklearn.metrics import mean_squared_error # MSE임
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

MSE = mean_squared_error(y1_test, y_predict)
print('MSE 는', MSE )

RMSE = RMSE(y1_test, y_predict)
print('RMSE 는', RMSE )


# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y1_test, y_predict)
print('R2는 ', r2)
