'''
acc = 98% 이상 !
'''
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 전처리
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = (x_train/255).reshape(-1, 28*28)
x_test = (x_test/255).reshape(-1, 28*28)

# 모델구성
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Dense(8, input_dim=(28*28))) 
model.add(Dense(16))
model.add(Dense(64))
model.add(Dense(82))
model.add(Dense(1260))
model.add(Dense(32))
model.add(Dense(10, activation='sigmoid')) #와... 뭐지?...

model.summary()

#3. 설명한 후 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) 
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# 걍 loss만 바꿔주면 되네?
model.fit(x_train,y_train, epochs=200, batch_size=900)  # 훨낫네.....ㅎㅎ 와 근데 미세하게 계속 올라가긴 한다잉~

#4. 평가와 예측
loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)

predict = model.predict(x_test)
# print(predict)
print(np.argmax(predict, axis = 1))

'''
# 아 분류에서 사실상 rmse, r2 볼 필요가 없는거 같은데?
from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE 는', RMSE(predict, y_test) )

# R2 구하기
from sklearn.metrics import r2_score 
r2 = r2_score(predict, y_test)
print('R2는 ', r2)
'''




