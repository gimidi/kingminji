# 한개의 모델에서 분류와 회귀를 동시에 나오게 할 수 있을까?
# 키(연속형), 성별(범주형)을 동시에 변수화
# train값도 두개로 주면 맞출까..? -> 아님

#1. 데이터
import numpy as np
x1_train = np.array([1,2,3,4,5,6,7,8,9,10])
x2_train = np.array([1,2,3,4,5,6,7,8,9,10])

y1_train = np.array([1,2,3,4,5,6,7,8,9,10])
y2_train = np.array([1,0,1,0,1,0,1,0,1,0])

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate

input1 = Input(shape=(1,))
x1 = Dense(100)(input1)
x1 = Dense(100)(x1)
x1 = Dense(100)(x1)

input2 = Input(shape=(1,))
x2 = Dense(100)(input2)
x2 = Dense(100)(x2)
x2 = Dense(100)(x2)

merge = concatenate([x1,x2])

# 아웃풋 1
x3 = Dense(50)(merge)
output1 = Dense(1)(x3)

# 아웃풋 2
x4 = Dense(70)(merge)
x4 = Dense(70)(x4)
output2 = Dense(1, activation='sigmoid')(x4)

model = Model(inputs=[input1,input2], outputs=[output1, output2])
model.summary()

# 컴파일, 훈련
model.compile(loss=['mse','binary_crossentropy'], optimizer='adam',
                metrics=['mse','acc'])
model.fit([x1_train,x2_train],[y1_train,y2_train], epochs=500)

# 평가 예측
loss = model.evaluate([x1_train,y2_train], [y1_train,y2_train])
print('loss: ',loss)

x1_pred = np.array([11,12,13,14])
x2_pred = np.array([11,12,13,14])

y_pred = model.predict([x1_pred,x2_pred])
print(y_pred)

