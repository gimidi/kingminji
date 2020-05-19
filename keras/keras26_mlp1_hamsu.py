# keras14_mlp를 Sequential에서 함수형으로 변경
# early_stopping 적용

#1. 데이터
import numpy as np
x = np.array([range(1,101), range(311,411), range(100)]).T  
y = np.array([range(101,201), range(711,811), range(100)]).T

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.23, shuffle=False
)

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
input1 = Input(shape=(3,))
dense1_1 = Dense(50, activation='relu')(input1) 
dense1_2 = Dense(80, activation='relu')(dense1_1)
dense1_3 = Dense(240, activation='relu')(dense1_2)
dense1_4 = Dense(1000, activation='relu')(dense1_3) 
dense1_5 = Dense(850, activation='relu')(dense1_4)
dense1_6 = Dense(526, activation='relu')(dense1_5)  
dense1_7 = Dense(250, activation='relu')(dense1_6) 
dense1_8 = Dense(3, activation='relu')(dense1_7)
model = Model(inputs=input1, outputs=dense1_8)
model.summary()

#3. 알아듣게 설명한 후 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
model.fit(x_train,y_train, epochs=10000, batch_size=5, callbacks=[early_stopping]
         ,validation_split = 0.23)
        

#4. 평가와 예측
loss,mse = model.evaluate(x_test,y_test, batch_size=5)
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
