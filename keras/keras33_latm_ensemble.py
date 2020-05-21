'''
                 행,         열,   몇개씩 자르는지
x의 shape = (batch_size, timesteps, feature)
input_shape = (timesteps, feature)
input_length = timesteps
input_dim = feature
'''
'''
loss: 1.1793e-05
[[79.843956]]
epochs=3000
model.add(LSTM(150)
LSTM param : 92400
'''
from numpy import array
from keras.models import Sequential, Model #keras의 씨퀀셜 모델로 하겠다
from keras.layers import Dense, LSTM, Input # Dense와 LSTM 레이어를 쓰겠다

#1. 데이터
x1 = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]]) # (13,3)
x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
            [50,60,70],[60,70,80],[70,80,90],[80,90,100],
            [90,100,110], [100,110,120],
            [2,3,4], [3,4,5], [4,5,6]]) # (13,3)            
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])  # (13,)
x1_predict = array([55,65,75]) 
x2_predict = array([65,75,85]) 

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)

x1_predict = x1_predict.reshape(1,3,1) 
x2_predict = x2_predict.reshape(1,3,1) 

# 함수형 모델 1
input1 = Input(shape=(3,1))
dense1_1 = LSTM(80, activation='relu')(input1) # input 레이어를 알려줘야 함/ 노드를 5개로 가지고 input1을 인풋레이어로 받는 댄스층
dense1_2 = Dense(32, activation='relu')(dense1_1)
dense1_5 = Dense(64, activation='relu')(dense1_2)

# 함수형 모델 2
input2 = Input(shape=(3,1))
dense2_1 = LSTM(80, activation='relu')(input2) # input 레이어를 알려줘야 함/ 노드를 5개로 가지고 input1을 인풋레이어로 받는 댄스층
dense2_2 = Dense(32, activation='relu')(dense2_1)
dense2_5 = Dense(64, activation='relu')(dense2_2)

# 합치기~
from keras.layers.merge import concatenate
merge1 = concatenate([dense1_5,dense2_5]) #output 자체를 묶어준다
middle1 = Dense(32)(merge1)
middle2 = Dense(32)(middle1)
output = Dense(1)(middle2)

model = Model(inputs=[input1, input2], outputs=output)
model.summary()

model.compile(optimizer='adam', loss='mse')
model.fit([x1,x2],y,epochs=3000)

y_predict = model.predict([x1_predict,x2_predict]) # 음.. 항상 list에 집어 넣어야 하는가
print('y_predict는 ',y_predict)



