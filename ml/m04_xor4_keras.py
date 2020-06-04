# ml 을 dl으로 바꾸기

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1. 데이터
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]
x_data = np.array(x_data)
y_data = np.array(y_data)
#2. 모델
# model = SVC()
# model = KNeighborsClassifier(n_neighbors=3) 

model = Sequential()
model.add(Dense(30, input_dim=2 ,activation='relu'))
model.add(Dense(60,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3. 실행
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, epochs=50, batch_size=1)
y_predict = model.predict(x_data)
print('y_predict는', y_predict)
loss, acc1 = model.evaluate(x_data, y_data)
print(acc1)
acc2 = accuracy_score([y_data, y_predict])
print(acc2)
