import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

wine = pd.read_csv('winequality-white.csv',sep=';')

x = np.array(wine.iloc[:,0:-1])
y = np.array(wine.iloc[:,-1])

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

from sklearn.preprocessing import OneHotEncoder
y = y.reshape(-1,1)
aaa = OneHotEncoder()
aaa.fit(y)
y = aaa.transform(y).toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(
    x, y, random_state=66, test_size=0.2 )

print(x_train.shape)
print(y_train.shape)

model = Sequential()
model.add(Dense(30, input_dim=11 ))
model.add(Dense(40))
model.add(Dense(120))
model.add(Dense(500,activation='relu'))
model.add(Dense(60))
model.add(Dense(32,activation='relu'))
model.add(Dense(7, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=300, batch_size=10, validation_split=0.2)
loss, acc = model.evaluate(x_test,y_test)
print('keras의 acc는',acc)
# score = model.score(x_test,y_test) 이건 아마 mL model에 들어있는거니까 없다고 인식하게찌?
# print('score는',score)
# print(np.argmax(a, axis = 1)+1)
