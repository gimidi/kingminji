#1. 데이터
import numpy as np
x = np.array(range(1,101))
y = np.array(range(101,201))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.23   
)

#2. 모델구성
from keras.models import Sequential 
from keras.layers import Dense 
model = Sequential()
model.add(Dense(5, input_dim=1)) 
model.add(Dense(8)) 
model.add(Dense(16)) 
model.add(Dense(32))
model.add(Dense(5))
model.add(Dense(1)) 

#3. 알아듣게 설명한 후 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train,y_train, epochs=200, batch_size=5,
         validation_split = 0.23)
# 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(12,10))
plt.subplot(2,1,1) # 2행 1열 에 첫번째꺼 그리겠다
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='gray', label='val_loss')
plt.grid()
plt.xlabel('epoch', size=15)
plt.ylabel('loss', size=15)
plt.legend(loc='upper left')

plt.subplot(2,1,2) # 2행 1열 에 두번째꺼 그리겠다
plt.plot(hist.history['acc'], marker='.', c='red', label='acc')  
plt.plot(hist.history['val_acc'], marker='.', c='gray', label='val_acc')
plt.ylabel('acc', size=15)
plt.xlabel('epoch', size=15)
plt.legend(loc='upper left')
# plt.legend(['acc', 'val acc']) # 걍 순서대로 들어가네
plt.show()        

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

