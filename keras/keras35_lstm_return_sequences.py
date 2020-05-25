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
input1 = Input(shape=(3,1)) # 근데 왜 얘는 shape=(,3,1) 이렇게 할 생각을 못했을까? 헷갈리는거 뻔히 알면서 그럴 필요가 잇나?/ 행무시 이거 하나 때문인가
dense1 = LSTM(10, return_sequences=True )(input1) # 원래 차원 그대로 아웃풋 하겠다 / 3차원 -> 3차원 그래도 
dense2 = LSTM(10)(dense1)
dense3 = Dense(5)(dense2)
dense4 = Dense(1)(dense3)

model = Model(inputs=input1, outputs=dense4)  # 함수형 모델임을 명시 레알 개신기하네...

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


# 여전히 난 주피터가 좋다.