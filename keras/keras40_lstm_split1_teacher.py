'''
보통 성능이 훅 좋아지는 구간이 없으면
층이랑 epo를 늘리는데
이건 뭐 엄청 늘려도 

mse 는 0.006379501428455114
[[8.980217]
 [9.88879 ]]
RMSE 는 0.07987178140516776
R2는  0.9744819941406604
이고

더 늘려도
성능이 떨어지는 사태가 벌어짐

내 생각에 문제는
데이터 차원을 
'''

import numpy as np

a = np.array(range(1,11))
size = 5
def split_x(seq, size) :
    aaa = []
    for i in range(len(seq) - size + 1) :
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa)) 
    return np.array(aaa)

dataset = split_x(a, size)
x = dataset[:,:4]
y = dataset[:,4]
# 이것만 봐도 데이터가 너무 적어서 dnn모델이 훨씬 성능이 높다는 것을 알 수 있음
# 대체 층도 더 쌓고 epo도 두배로 넣었는데 성능이 더 안좋아지는 경우는 뭐임?
# lstm 왜 쓰는겨.....?????

x = x.reshape(x.shape[0],x.shape[1],1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.23, shuffle=False
)

#2. 모델구성
from keras.models import Sequential 
from keras.layers import LSTM, Dense 

model = Sequential()
model.add(LSTM(320, input_shape=(4,1))) # input을 넣는거야 무조권 ^_^ 
model.add(Dense(32))
model.add(Dense(640))
model.add(Dense(640))
model.add(Dense(16))  
model.add(Dense(1)) 

#3. 설명한 후 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto') # 어느정도 사이즈부터 성능향상에 도움을 주는지 아직 머르겠씁니다
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train, epochs=10000, batch_size=3, callbacks=[early_stopping])  

#4. 평가와 예측
loss,mse = model.evaluate(x_test,y_test) 
print('mse 는',mse)
y_predict = model.predict(x_test)
print(y_predict)

from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE 는', RMSE(y_test, y_predict) )

# R2 구하기
from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)
print('R2는 ', r2)
