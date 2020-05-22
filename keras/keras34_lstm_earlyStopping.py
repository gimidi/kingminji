'''
                 행,         열,   몇개씩 자르는지
x의 shape = (batch_size, timesteps, feature)
input_shape = (timesteps, feature)
input_length = timesteps
input_dim = feature
'''
'''
함수형
loss: 4.0469e-05
[[80.539894]]
epochs=4000
model.add(LSTM(150)
LSTM param : 92400
'''
from numpy import array
from keras.models import Sequential, Model #keras의 씨퀀셜 모델로 하겠다
from keras.layers import Dense, LSTM, Input # Dense와 LSTM 레이어를 쓰겠다

#1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]]) # (13,3)
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])  # (13,)
x_predict = array([50,60,70]) 

x = x.reshape(x.shape[0], x.shape[1], 1)
x_predict = x_predict.reshape(1,3,1) 

# 함수형 모델
input1 = Input(shape=(3,1))
dense1 = LSTM(150, activation='relu')(input1) # input 레이어를 알려줘야 함/ 노드를 5개로 가지고 input1을 인풋레이어로 받는 댄스층
dense2 = Dense(32, activation='relu')(dense1)
dense3 = Dense(64, activation='relu')(dense2)
dense4 = Dense(128, activation='relu')(dense3)
dense5 = Dense(64, activation='relu')(dense4)
output1 = Dense(1)(dense5)

model = Model(inputs=input1, outputs=output1)  # 함수형 모델임을 명시 레알 개신기하네...

model.summary()

# 2. 모델구성
model = Sequential()
# model.add(LSTM(10, activation='relu', input_shape=(3,1))) 
model.add(LSTM(150, activation='relu', input_length=3, input_dim=1))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1))
model.summary()

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto') # 난 머르겠다
model.compile(optimizer='adam', loss='mse')
model.fit(x,y,epochs=3000, callbacks=[early_stopping])

y_predict = model.predict(x_predict)
print(y_predict)


