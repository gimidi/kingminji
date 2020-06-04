<<<<<<< HEAD
'''
acc는 0.9912280440330505
'''
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()

x = dataset.data
y = dataset.target

# 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

from keras.utils import np_utils
y = np_utils.to_categorical(y)

# 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.2 )

# 모델링 
import numpy as np
from keras.models import Sequential, Model 
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense,LSTM

# cnn
x_train = x_train.reshape(-1,2,3,5)
x_test = x_test.reshape(-1,2,3,5)

model = Sequential()
model.add(Conv2D(16, (2,2),input_shape=(2,3,5), padding='same'))
model.add(Conv2D(16, (2,2), padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(64, (2,2), padding='same'))
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(2, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=50, batch_size=5, verbose=2, validation_split=0.2)

loss, acc = model.evaluate(x_test, y_test)
print('acc는', acc)

# 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(12,10))
plt.subplot(2,1,1) 
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='gray', label='val_loss')
plt.grid()
plt.xlabel('epoch', size=15)
plt.ylabel('loss', size=15)
plt.legend(loc='upper left')

plt.subplot(2,1,2)
plt.plot(hist.history['acc'], marker='.', c='red', label='acc')  
plt.plot(hist.history['val_acc'], marker='.', c='gray', label='val_acc')
plt.ylabel('acc', size=15)
plt.xlabel('epoch', size=15)
plt.legend(loc='upper left')
=======
'''
acc는 0.9912280440330505
'''
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()

x = dataset.data
y = dataset.target

# 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

from keras.utils import np_utils
y = np_utils.to_categorical(y)

# 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.2 )

# 모델링 
import numpy as np
from keras.models import Sequential, Model 
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense,LSTM

# cnn
x_train = x_train.reshape(-1,2,3,5)
x_test = x_test.reshape(-1,2,3,5)

model = Sequential()
model.add(Conv2D(16, (2,2),input_shape=(2,3,5), padding='same'))
model.add(Conv2D(16, (2,2), padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(64, (2,2), padding='same'))
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(2, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=50, batch_size=5, verbose=2, validation_split=0.2)

loss, acc = model.evaluate(x_test, y_test)
print('acc는', acc)

# 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(12,10))
plt.subplot(2,1,1) 
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='gray', label='val_loss')
plt.grid()
plt.xlabel('epoch', size=15)
plt.ylabel('loss', size=15)
plt.legend(loc='upper left')

plt.subplot(2,1,2)
plt.plot(hist.history['acc'], marker='.', c='red', label='acc')  
plt.plot(hist.history['val_acc'], marker='.', c='gray', label='val_acc')
plt.ylabel('acc', size=15)
plt.xlabel('epoch', size=15)
plt.legend(loc='upper left')
>>>>>>> fa2732da2f961158aee21e0eecf6dd0dd4f77931
plt.show()