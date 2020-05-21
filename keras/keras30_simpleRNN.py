'''
epochs=3000 -> 2000에
 loss: 4.1564e-08 -> 2.6722e-08
[[[5]
  [6]
  [7]]]
[[7.9972777]] -> [[8.029411]] 
model.add(LSTM(150))
LSTM param : 92400

model.add(SimpleRNN(150))
SimpleRNN param : 22800
'''
from numpy import array
from keras.models import Sequential #keras의 씨퀀셜 모델로 하겠다
from keras.layers import Dense, SimpleRNN # Dense와 LSTM 레이어를 쓰겠다

#1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]]) # (4,3)
y = array([4,5,6,7]) # ( ,4)
y1 = array([[4,5,6,7]]) # (1,4)
y2 = array([[4],[5],[6],[7]])  # (4,1)
print('x shape는',x.shape) # (4, 3)
print('y shape는',y.shape) # (4,)
print('y1 shape는',y1.shape) # (1,4)
print('y2 shape는',y2.shape) # (1,4)

# input_dim = 1 -> 1차원이다
# x = x.reshape(4, 3, 1)
# reshape할때 모든요소 곱해서 같으면 잘 한거다.
x = x.reshape(x.shape[0], x.shape[1], 1)

print('reshape한 x는',x)
print('x shape는',x.shape) # (4, 3, 1) -> 연속된 데이터에 대해서는 하나씩 작업하겠다. 한개씩 빼고는 4행 3열짜리 배열이다. 라고 생각하면 된다.
# 이거 해주는 이유는 lstm 이 3차원 데이터를 원하기 때문임
# 2. 모델구성
model = Sequential()
model.add(SimpleRNN(150, activation='relu', input_shape=(3,1))) 
model.add(Dense(32))
model.add(Dense(640))
model.add(Dense(126))
model.add(Dense(20))
model.add(Dense(64))

model.add(Dense(1))

model.summary()
'''
model.compile(optimizer='adam', loss='mse')
model.fit(x,y,epochs=2000)

x_input = array([5,6,7]) # 도출되는 y형태는 (3, ) 처럼 스칼라임 / 
x_input = x_input.reshape(1,3,1) # 아 전부 3차원 텐서로 바꿔주나보네??/ 아 이거 넣는값이라서 x랑 똑같이 바꿔주는거자너 뭘 다른거라고 말하고 있니..??
# input이니까 x데이터 모양으로 바꿔준다는거야 앙상블에서 계속 배웠자너
# input데이터는 x데이터 대로만(3차원
# ) 넣으면 되고 결과값은 어찌됏든 y데이터 형식(스칼라)으로 나온다!
print(x_input)

yhat = model.predict(x_input)
print(yhat) # 8 나와야 허는디?
# -> LSTM을 쓰기에 너무 적은 데이터다
# LSTM은 만들어져있는거고 우리가 튜닝하는 부분은 batch_size (fit, dense에도 적용이 됐었어?????이게???), 노드, 레이어
'''


