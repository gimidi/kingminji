from keras.datasets import cifar10, mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Dropout, MaxPool2D, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2
import numpy as np
from keras.applications import VGG16, VGG19, Xception, ResNet101, ResNet101V2, ResNet152
from keras.applications import ResNet152V2, ResNet50, ResNet50V2, InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2, DenseNet121, DenseNet169, DenseNet201
from keras.applications import NASNetLarge, NASNetMobile
from keras.preprocessing.image import img_to_array, image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation
from keras.optimizers import Adam

inceptionResnet2 = InceptionResNetV2()

inception = inceptionResnet2(
    weights=None, 
    include_top=True, 
    classes=10,
    input_shape=(32,32,3)
)

(x_train, y_train),(x_test,y_test) = cifar10.load_data()
# 전처리 (min_max스케일링, 원-핫 인코딩)
x_train = (x_train/255)
x_test = (x_test/255)

model = Sequential()
model.add(inception)
# model.add(Flatten())
model.add(Dense(256))
model.add(Dense(256))
model.add(Dense(10, activation='softmax'))
model.summary()


model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.0004), metrics=['acc'])
hist = model.fit(x_train, y_train,
                epochs=10, batch_size=32, verbose=1,
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
