#1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x2 = np.array([11,12,13])
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

#3. 훈련 / 어떤 접근을 사용했는가 -> 성능평가 방식으로 loss설정과 같은 mse를 설정하였음
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x,y, epochs=50, batch_size=3)  # Dense보다 epochs가  중요한듯

#4. 평가와 예측
loss, acc = model.evaluate(x,y)
print('loss 는',loss)
print('acc 는', acc)

