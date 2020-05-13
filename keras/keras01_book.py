# x,y

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
# 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
# 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(1, activation = 'relu'))
# 훈련
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
model.fit(x,y, epochs=200, batch_size=1)
# 평가
loss, acc = model.evaluate(x, y, batch_size=1)

print("loss :", loss)
print("acc :", acc)