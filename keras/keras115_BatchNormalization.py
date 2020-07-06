# 배치 노멀리제이션(일반화) minmax scalar이 노멀라이제이션임
# 출력값(w)을 일반화 (0~1사이로 수렴)
# 과적합 피하기 2.



# regularizer(레귤러라리제이션_규제화)
'''
L1규제 : 가중치의 절대값 합
regularizer.l1(l=0.01)

L2규제 : 가중치의 제곱 합
regularizer.l2(l=0.01)

loss = L1 * reduce_sum(abs(x))
loss = L2 * reduce_sum(square(x))
(레이어에만 사용가능! 나는 maxpooling에 사용해서 안돌아갔었다..!)

기존의 성능
loss: 0.0422 - acc: 0.9862 - val_loss: 1.8259 - val_acc: 0.7427

 ((((((W 관리))))))
l1규제 사용시(0.001)
loss: 0.4874 - acc: 0.9519 - val_loss: 1.3270 - val_acc: 0.7535

l1_l2규제 사용시(0.001)
loss: 1.5826 - acc: 0.4898 - val_loss: 1.6306 - val_acc: 0.4723 ....????

l2규제 사용시(0.001)
loss: 0.4915 - acc: 0.9527 - val_loss: 1.3829 - val_acc: 0.7447

BatchNormalization 사용시(layer마다 적용)
loss: 0.0504 - acc: 0.9836 - val_loss: 1.0179 - val_acc: 0.7732

 ((((((노드관리))))))
규제 빼고 dropout 6개(layer마다) 사용시
loss: 0.2913 - acc: 0.8946 - val_loss: 0.7143 - val_acc: 0.7885

규제 빼고 dropout 3개(maxpooling다음 layer에만) 사용시 ㄷㄷㄷ
loss: 0.1655 - acc: 0.9411 - val_loss: 0.8296 - val_acc: 0.7927

=> 결론 l1과 BatchNormalization과 (maxpooling다음 layer에만) dropout을 쓴다 !

'''
'''
acc = 83 %
'''
from keras.datasets import cifar10, mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Dropout, MaxPool2D, Flatten, MaxPooling2D, BatchNormalization, Activation
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2
import numpy as np
(x_train, y_train),(x_test,y_test) = cifar10.load_data()
# 전처리 (min_max스케일링, 원-핫 인코딩)
x_train = (x_train/255)
x_test = (x_test/255)
# CNN
model = Sequential()
model.add(Conv2D(32, kernel_size=3, padding='same', input_shape=(32, 32, 3), kernel_regularizer=l1(0.001)))
model.add(BatchNormalization()) # activation이 적용되기 전에 써주는거래 ㄷㄷ, activation전에 쓰려고 만든거래
model.add(Activation('relu'))
model.add(Conv2D(32, kernel_size=3, padding='same', kernel_regularizer=l1(0.001)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

model.add(Conv2D(64, kernel_size=3, padding='same', kernel_regularizer=l1(0.001)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=3, padding='same', kernel_regularizer=l1(0.001)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

model.add(Conv2D(128, kernel_size=3, padding='same', kernel_regularizer=l1(0.001)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, kernel_size=3, padding='same', kernel_regularizer=l1(0.001)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

model.add(Flatten())
model.add(Dense(256, kernel_regularizer=l1(0.001)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.0004), metrics=['acc'])
hist = model.fit(x_train, y_train,
                epochs=20, batch_size=32, verbose=1,
                validation_split=0.3) 
##### plt 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(9,5))

# 1
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

# 2
plt.subplot(2,1,2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()

# 평가 및 예측
loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)

predict = model.predict(x_test)
print(predict)



