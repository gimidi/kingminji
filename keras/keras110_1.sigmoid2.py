# 한개의 모델에서 분류와 회귀를 동시에 나오게 할 수 있을까?
# 키(연속형), 성별(범주형)을 동시에 변수화
# train값도 두개로 주면 맞출까..? -> 아님

#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([0,1,0,1,0,1,0,1,0,1])

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate

model = Sequential()
model.add(Dense(100, input_shape=(1,), activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# 컴파일, 훈련
model.compile(loss=['binary_crossentropy'], optimizer='adam',
                metrics=['acc'])
model.fit(x_train,y_train, epochs=500)

# 평가 예측
loss = model.evaluate(x_train,y_train)
print('loss: ',loss)

x1_pred = np.array([11,12,13,14])

y_pred = model.predict(x1_pred)
print(y_pred)

