'''
                 행,         열,   몇개씩 자르는지
x의 shape = (batch_size, timesteps, feature)
input_shape = (timesteps, feature)
input_length = timesteps
input_dim = feature
'''
'''
epochs=3000 -> 1500 
loss: 1.1793e-05 -> 1.9278e-05
[[79.843956]] -> [[80.00561]]
model.add(LSTM(150)
LSTM param : 92400 

model.add(SimpleRNN(150))
SimpleRNN param : 22800
'''


from numpy import array
from keras.models import Sequential #keras의 씨퀀셜 모델로 하겠다
from keras.layers import Dense, SimpleRNN # Dense와 LSTM 레이어를 쓰겠다

#1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]]) # (13,3)
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])  # (13,)
x_predict = array([50,60,70]) 

x = x.reshape(x.shape[0], x.shape[1], 1)
x_predict = x_predict.reshape(1,3,1) 

# 2. 모델구성
model = Sequential()
# model.add(LSTM(10, activation='relu', input_shape=(3,1))) 
model.add(SimpleRNN(150, activation='relu', input_length=3, input_dim=1))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1))
model.summary()

model.compile(optimizer='adam', loss='mse')
model.fit(x,y,epochs=1500)

y_predict = model.predict(x_predict)
print(y_predict)


