import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])  # 10개의 1 꾸러미네 -> ndim = 1
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([101,102,103,104,105,106,107,108,109,110])

from keras.models import Sequential
from keras.layers import Dense

# 모델만들기
model = Sequential()
model.add(Dense(1, input_dim=1, activation='relu')) # 입력층
model.add(Dense(8)) # 히든레이어
model.add(Dense(1, activation = 'sigmoid')) # 출력층

# 훈련시키기
model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train, epochs=500, batch_size=2, validation_data = (x_train, y_train))

# 평가
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("loss :", loss)
print("acc :", acc)