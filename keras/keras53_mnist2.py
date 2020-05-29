'''
softmax _ binary -> 이게 더 빨리 올라가긴 함
sigmoid _ binary
acc 는 0.982599139213562
'''
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# plt.imshow(x_train[59899],'gray')
# # plt.imshow(x_train[0])
# plt.show()

print(x_train[0]) #[0,0,0,0,....................0,0,.....]
print('y_train[0]:',y_train[0]) # 5
print(x_train[0].shape) # (28, 28) 짜리
#----------------------------------------------데이터보기

# 순자가 0 ~ 9 중 어느건지 알아맞히고 싶다 (분류)
# 원핫 인코딩부터 해주기
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) # (60000, 10)

# x가 너무 많으니까(0~255까지) x도 전처리 해줄 것임
# CNN 이니까 4차원으로 바꿔준다
# minmax로 (0~1) 숫자로 바꿔줌 -> 연산하기에 편함
# x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
# x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255
x_train = (x_train/255).reshape(60000, 28, 28, 1)
x_test = (x_test/255).reshape(10000, 28, 28, 1)
# x_train = x_train / 255 # 걍 최솟값이 0이니까 최댓값으로 나누기만 해도 됨

# 모델구성
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(1, (2,2), input_shape=(28,28,1), padding='same')) 
# model.add(Conv2D(32, (3,3), padding='same')) 
# model.add(Conv2D(64, (2,2), padding='same')) 
# model.add(Conv2D(16, (2,2), padding='same'))
model.add(Conv2D(5, (2,2), padding='same')) 
# model.add(Conv2D(5, (2,2), strides=2))  # (3,3,5) 6/2=3
# model.add(Conv2D(5, (2,2), strides=2, padding='same'))  # (3,3,5) 6/2=3
# model.add(MaxPooling2D(pool_size=2)) # (None, 3, 3, 5)
# 4차원을 2차원으로 쫙 펴준다
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

#3. 설명한 후 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])  # 성능 훨씬 좋음 -> 사실상 binary를 사용하는게 맞다는거임
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# 걍 loss만 바꿔주면 되네?
model.fit(x_train,y_train, epochs=50, batch_size=500) #  배치도 조정할것 <- 뭔지를 알아야 더 조정할텐데

#4. 평가와 예측
loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)

predict = model.predict(x_test)
print(predict)
print(np.argmax(predict, axis = 1))




