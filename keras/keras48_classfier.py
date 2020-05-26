# 과제 해야함

import numpy as np

x = np.array(range(1,11))
y = np.array([1,0,1,0,1,0,1,0,1,0]) # 10개~

# 모델 구성하기

from keras.models import Sequential 
from keras.layers import LSTM, Dense # 하지만 난 Dense 쓸거야

model = Sequential()
model.add(Dense(16, input_dim=(1), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu')) 
model.add(Dense(64, activation='relu')) 
model.add(Dense(1, activation='sigmoid'))
#3. 설명한 후 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# 걍 loss만 바꿔주면 되네?
model.fit(x,y, epochs=800, batch_size=1)  

a = model.predict([1,2,3])
predict = [round(x) for x in a.reshape(3) ]
print(predict)

'''
#4. 평가와 예측
loss,acc = model.evaluate(x,y) 
print('loss 는',loss)
print('acc 는', acc)

x_pred = np.array([1,2,3])
x_pred_답 =  np.array([1,0,1])
y_pred = model.predict(x_pred)
print('y_pred 는',y_pred)


from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE 는', RMSE(x_pred_답, y_pred) )

# R2 구하기
from sklearn.metrics import r2_score 
r2 = r2_score(x_pred_답, y_pred)
print('R2는 ', r2)
'''