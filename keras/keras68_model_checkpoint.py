<<<<<<< HEAD
'''

'''
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = (x_train/255).reshape(60000, 28, 28, 1)
x_test = (x_test/255).reshape(10000, 28, 28, 1)

# 모델구성
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout,MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Conv2D(16, (2,2), input_shape=(28,28,1), padding='same')) 
model.add(Conv2D(32, (2,2), padding='same')) 
model.add(Conv2D(32, (2,2), padding='same')) 
model.add(MaxPooling2D(2))
model.add(Dropout(0.2))
model.add(Conv2D(43, (2,2), padding='same')) 
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

# 훈련
early_stop = EarlyStopping(monitor='loss', patience=100, mode='auto')
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                        save_best_only=True) # 좋은것만 저장하겠다
# 터미널에서 보이는데 굳이 저장하는 이유는? 그냥 눈으로 보려고 그러는거야..? 용량은 어케해?
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])  # 성능 훨씬 좋음 -> 사실상 binary를 사용하는게 맞다는거임
hist = model.fit(x_train,y_train, epochs=20, batch_size=500, validation_split=0.2,
                callbacks = [early_stop, checkpoint]) #  배치도 조정할것 <- 뭔지를 알아야 더 조정할텐데


import matplotlib.pyplot as plt
plt.figure(figsize=(10,6)) # 가로, 세로
# 두개 그릴거면 subplot 써야함
plt.subplot(2,1,1) # 2행 1열 에 첫번째꺼 그리겠다
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='gray', label='val_loss')
plt.grid()
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')

plt.subplot(2,1,2) # 2행 1열 에 두번째꺼 그리겠다
plt.plot(hist.history['acc'])  
plt.plot(hist.history['val_acc'])
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val acc']) # 걍 순서대로 들어가네
plt.show()

# 평가 및 예측
loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)

predict = model.predict(x_test)
print(predict)
print(np.argmax(predict, axis = 1))




=======
'''

'''
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = (x_train/255).reshape(60000, 28, 28, 1)
x_test = (x_test/255).reshape(10000, 28, 28, 1)

# 모델구성
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout,MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Conv2D(16, (2,2), input_shape=(28,28,1), padding='same')) 
model.add(Conv2D(32, (2,2), padding='same')) 
model.add(Conv2D(32, (2,2), padding='same')) 
model.add(MaxPooling2D(2))
model.add(Dropout(0.2))
model.add(Conv2D(43, (2,2), padding='same')) 
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

# 훈련
early_stop = EarlyStopping(monitor='loss', patience=100, mode='auto')
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                        save_best_only=True) # 좋은것만 저장하겠다
# 터미널에서 보이는데 굳이 저장하는 이유는? 그냥 눈으로 보려고 그러는거야..? 용량은 어케해?
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])  # 성능 훨씬 좋음 -> 사실상 binary를 사용하는게 맞다는거임
hist = model.fit(x_train,y_train, epochs=20, batch_size=500, validation_split=0.2,
                callbacks = [early_stop, checkpoint]) #  배치도 조정할것 <- 뭔지를 알아야 더 조정할텐데


import matplotlib.pyplot as plt
plt.figure(figsize=(10,6)) # 가로, 세로
# 두개 그릴거면 subplot 써야함
plt.subplot(2,1,1) # 2행 1열 에 첫번째꺼 그리겠다
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='gray', label='val_loss')
plt.grid()
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')

plt.subplot(2,1,2) # 2행 1열 에 두번째꺼 그리겠다
plt.plot(hist.history['acc'])  
plt.plot(hist.history['val_acc'])
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val acc']) # 걍 순서대로 들어가네
plt.show()

# 평가 및 예측
loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)

predict = model.predict(x_test)
print(predict)
print(np.argmax(predict, axis = 1))




>>>>>>> fa2732da2f961158aee21e0eecf6dd0dd4f77931
