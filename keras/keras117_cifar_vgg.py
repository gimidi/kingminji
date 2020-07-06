
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

l1규제 사용시
loss: 0.4874 - acc: 0.9519 - val_loss: 1.3270 - val_acc: 0.7535

규제 빼고 dropout 6개(layer마다) 사용시
loss: 0.2913 - acc: 0.8946 - val_loss: 0.7143 - val_acc: 0.7885

규제 빼고 dropout 3개(maxpooling다음 layer에만) 사용시 ㄷㄷㄷ
loss: 0.1655 - acc: 0.9411 - val_loss: 0.8296 - val_acc: 0.7927

vgg 사용시

'''
from keras.datasets import cifar10, mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Dropout, MaxPool2D, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2
import numpy as np
from keras.applications import VGG16, VGG19
from keras.preprocessing.image import img_to_array, image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation
from keras.optimizers import Adam

vgg16 = VGG16(
    weights= 'imagenet', 
    include_top=Ture, 
    classes=10,
    input_shape=(32,32,3)
)

vgg19 = VGG19(
    weights=None, 
    include_top=True, 
    classes=10,
    input_shape=(32,32,3)
)

(x_train, y_train),(x_test,y_test) = cifar10.load_data()
# 전처리 (min_max스케일링, 원-핫 인코딩)
x_train = (x_train/255)
x_test = (x_test/255)

# vgg
model = Sequential()
model.add(vgg16)
# model.add(Flatten())
model.add(Dense(256))
model.add(Dense(256))
model.add(Dense(256))
model.add(Dense(256))
model.add(Dense(256))
model.add(Dense(256))
model.add(Dense(10, activation='softmax'))
model.summary()

# CNN
# model = Sequential()
# model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(32, 32, 3)))
# # model.add(Dropout(0.2))
# model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))
# model.add(Dropout(0.2))

# model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
# # model.add(Dropout(0.2))
# model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))
# model.add(Dropout(0.2))

# model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
# # model.add(Dropout(0.2))
# model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))
# model.add(Dropout(0.2))

# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.0004), metrics=['acc'])
hist = model.fit(x_train, y_train,
                epochs=20, batch_size=32, verbose=1,
                validation_split=0.3) 
# 그리자
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6)) # 가로, 세로
# 두개 그릴거면 subplot 써야함
plt.subplot(2,1,1) # 2행 1열 에 첫번째꺼 그리겠다
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='black', label='val_loss')
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



