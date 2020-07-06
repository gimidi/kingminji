'''
acc = 83 %
'''
from keras.datasets import cifar10, mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Dropout, MaxPool2D, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
from keras.optimizers import Adam

(x_train, y_train),(x_test,y_test) = cifar10.load_data()
# 전처리 (min_max스케일링, 원-핫 인코딩)
x_train = (x_train/255)
x_test = (x_test/255)
from keras.utils import np_utils
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
print(y_train.shape)
# CNN
# model = Sequential()
# model.add(Conv2D(16, kernel_size = (2,2), input_shape=(32,32,3), padding='same',activation ='relu'))
# model.add(Conv2D(16, kernel_size = (5,5), padding = 'Same',activation ='relu'))
# model.add(MaxPool2D(pool_size=(2,2), strides=2, padding='same'))
# model.add(Dropout(0.25))

# model.add(Conv2D(32, (3,3), padding='same'))
# model.add(Conv2D(32, kernel_size = (5,5), padding = 'same',activation ='relu'))
# model.add(MaxPool2D(pool_size=(2,2), strides=2, padding='same'))
# model.add(Dropout(0.2))
# # model.add(Conv2D(32, (2,2), padding='same',activation ='relu'))
# # model.add(Conv2D(32, (2,2), padding='same',activation ='relu'))
# # model.add(MaxPooling2D(2,2)) 
# # model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(10, activation='softmax'))
# model.summary()
model = Sequential()
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.0004), metrics=['acc'])
hist = model.fit(x_train, y_train,
                epochs=30, batch_size=32, verbose=1,
                validation_split=0.3) 
# 그리자
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6)) # 가로, 세로
# 두개 그릴거면 subplot 써야함
plt.subplot(2,1,1) # 2행 1열 에 첫번째꺼 그리겠다
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='red', label='val_loss')
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



