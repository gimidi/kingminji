'''
acc: 0.9964
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
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input

input1 = Input(shape=(28,28,1))
layer1 = Conv2D(16, (2,2), padding='same')(input1) # input 레이어를 알려줘야 함/ 노드를 5개로 가지고 input1을 인풋레이어로 받는 댄스층
layer2 = Dropout(0.5)(layer1)
layer3 = Conv2D(32, (3,3), padding='same')(layer2)
layer4 = Dropout(0.2)(layer3)
layer5 = Conv2D(32, (3,3), padding='same')(layer4)
layer6 = MaxPooling2D(2,2)(layer5)
layer7 = Dropout(0.2)(layer6)
layer8 = Flatten()(layer7)
output1 = Dense(10, activation='softmax')(layer8)
model = Model(inputs=input1, outputs=output1)
model.summary()

#3. 설명한 후 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])  # 성능 훨씬 좋음 -> 사실상 binary를 사용하는게 맞다는거임

# 걍 loss만 바꿔주면 되네?
model.fit(x_train,y_train, epochs=15, batch_size=500)  # 훨낫네.....ㅎㅎ 와 근데 미세하게 계속 올라가긴 한다잉~

#4. 평가와 예측
loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)





