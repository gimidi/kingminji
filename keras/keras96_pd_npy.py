<<<<<<< HEAD
# 95번을 불러와서 모델을 완성하시오

'''
acc는 1.0
'''
import numpy as np
dataset = np.load('./data/iris.npy')

x = dataset[:,:-1]
y = dataset[:,-1]

# 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x) 

from keras.utils import np_utils
y = np_utils.to_categorical(y)

# 모델링
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.2 )

import numpy as np
from keras.models import Sequential, Model 
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model = Sequential()
model.add(Dense(16, input_shape=(4,)))
model.add(Dense(16))
model.add(Dropout(0.2))

model.add(Dense(64))
model.add(Dense(64))
model.add(Dropout(0.2))

model.add(Dense(3, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=60, batch_size=2, verbose=2, validation_split=0.2)

loss, acc = model.evaluate(x_test, y_test)
print('acc는', acc)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,6)) 
plt.subplot(2,1,1) 
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='gray', label='val_loss')
plt.grid()
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')

plt.subplot(2,1,2)
plt.plot(hist.history['acc'])  
plt.plot(hist.history['val_acc'])
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val acc']) 
=======
# 95번을 불러와서 모델을 완성하시오

'''
acc는 1.0
'''
import numpy as np
dataset = np.load('./data/iris.npy')

x = dataset[:,:-1]
y = dataset[:,-1]

# 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x) 

from keras.utils import np_utils
y = np_utils.to_categorical(y)

# 모델링
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.2 )

import numpy as np
from keras.models import Sequential, Model 
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model = Sequential()
model.add(Dense(16, input_shape=(4,)))
model.add(Dense(16))
model.add(Dropout(0.2))

model.add(Dense(64))
model.add(Dense(64))
model.add(Dropout(0.2))

model.add(Dense(3, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=60, batch_size=2, verbose=2, validation_split=0.2)

loss, acc = model.evaluate(x_test, y_test)
print('acc는', acc)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,6)) 
plt.subplot(2,1,1) 
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='gray', label='val_loss')
plt.grid()
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')

plt.subplot(2,1,2)
plt.plot(hist.history['acc'])  
plt.plot(hist.history['val_acc'])
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val acc']) 
>>>>>>> fa2732da2f961158aee21e0eecf6dd0dd4f77931
plt.show()