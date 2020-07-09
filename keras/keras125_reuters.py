from keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test,y_test) = reuters.load_data(num_words=1000, test_split=0.2) # 가장 많이 쓰여진 단어 1000개

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

print(x_train[0])
print(y_train[0])

print(len(x_train[0]))

### y의 카테고리 개수 출력
category = np.max(y_train) + 1 
print('카테고리:',category) # 46 따라서 46가지로 구성되어 있음

### y의 유니크한 값 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)  # 0부터 45 있다

y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count() # 주간과제 : 판다스의 그룹바이 공부해보세유

print(bbb)
print(bbb.shape)

# 와꾸를 예뿌게 맞추자 # 패딩패딩!!

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, maxlen=100, padding='pre') # truncating=) # 짤라버리겠다
x_test = pad_sequences(x_test, maxlen=100, padding='pre')
# 46개의 다중분류임
# print(len(x_train[0]))
# print(len(x_train[-1]))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape) # (8982, 100)
print(x_test.shape) # (2246, 100)

#2. 모델
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Flatten

model = Sequential()
# model.add(Embedding(1000, 100, input_length=100)) # 와꾸 다시 정리해야할듯
model.add(Embedding(1000, 100))
model.add(LSTM(64))
model.add(Dense(46, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(x_train, y_train, batch_size=100, epochs=10,
                    validation_split=0.2)

acc = model.evaluate(x_test, y_test)[1]
print("acc :", acc)

y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker='.', c='red', label='TestSet Loss')
plt.plot(y_loss, marker='.', c='black', label='TrainSet Loss')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()