'''

'''
import numpy as np
from keras.models import Sequential 
from keras.layers import LSTM, Dense 

a = np.array(range(1,101))
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

x = x.reshape(x.shape[0],x.shape[1],1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.23, shuffle=False
)

#2. 모델 구성
from keras.models import load_model
model = load_model('./model/save44.h5')
model.add(Dense(1, name='mm'))
model.summary() 

#3. 설명한 후 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto') # 어느정도 사이즈부터 성능향상에 도움을 주는지 아직 머르겠씁니다
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train,y_train, epochs=10, batch_size=5, validation_split=0.2,
                 callbacks=[early_stopping])    # 걍 이렇게 넣어도 훈련 시키네용...!! ㅎㅎ
print(hist.history) 
# 딕셔너리 형태 _ {'loss': [2016, 970, 413], 'mse': [2016, 970, 413]}
print(hist.history.keys())
# dict_keys(['loss', 'mse'])
print(hist.history.values())
# dict_values([[2022.6484224502356, 1114.6739372357931, 463.4480189493258], [2022.6486, 1114.674, 463.448]])
print(hist) 
# <keras.callbacks.callbacks.History object at 0x000002416B5DD448> 얜 왜 안보여주는겨.....??

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.plot(hist.history['acc'])   # epochs를 어떻게 자동으로 인식하고 있는거지.. ㄷㄷㄷㄷ/ 그게 아니고 1부터 입력 되는것임
plt.plot(hist.history['val_acc'])

plt.title('loss')
plt.ylabel('evaluate')
plt.xlabel('epoch')
plt.legend(['train loss', 'train_val loss', 'train acc', 'train_val acc']) # 걍 순서대로 들어가네
plt.show()

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
