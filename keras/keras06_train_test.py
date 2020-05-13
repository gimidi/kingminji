#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) # 문제집
y_train = np.array([1,2,3,4,5,6,7,8,9,10]) # 문제집답지
x_test = np.array([11,12,13,14,15]) # 모의고사 문제
y_test = np.array([11,12,13,14,15]) # 모의고사 답지
x_pred = np.array([16, 17, 18]) # 실제시험
# shift + del 열삭제/ 커서놓고 cntrol+c 하면 열 복사가 됨

#2. 모델구성 / 튜닝
from keras.models import Sequential 
from keras.layers import Dense # DNN 구조의 가장 베이스가 되는 모델
model = Sequential()
model.add(Dense(5, input_dim=1)) #인풋레이어/ 5개의 노드 아웃풋/ # 10개짜리 한 덩어리 input하겠다
model.add(Dense(3)) # 인풋이 5, 아웃풋이 3개 노드인 아웃풋 히든레이어
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(1)) # 아웃레이어

#3. 훈련 / 어떤 접근을 사용했는가
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x,y, epochs=200, batch_size=3)  # Dense보다 epochs가  중요한듯

#4. 성능평가
mse, mse = model.evaluate(x_test,y_test)
print('mse 는',mse)

# 예측해보깅
model.predict([10,15,13,100,105,108])

# e-15만들기 챌린지 해보았지만 결국 e-12가 최고였다고 한다.. -> 5개중에 3개 딱떨어지게 도출됨
