'''
tensorboard = TensorBoard(log_dir='graph', histogram_freq=0,write_graph=True, write_images=True) 
hist = model.fit(x_train,y_train, epochs=20, batch_size=5, validation_split=0.2,callbacks=[early_stopping, tensorboard]) 
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

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.23, shuffle=False
)

#2. 모델 구성
from keras.models import load_model
model = Sequential()
model.add(Dense(5, input_dim=(4)))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))
model.summary()

#3. 설명한 후 훈련
from keras.callbacks import EarlyStopping, TensorBoard
tensorboard = TensorBoard(log_dir='graph', histogram_freq=0, 
                    write_graph=True, write_images=True) 

early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto') # 어느정도 사이즈부터 성능향상에 도움을 주는지 아직 머르겠씁니다
model.compile(loss='mse', optimizer='adam', metrics=['acc']) # 걍 반대로 올라가는거 보여주려고
hist = model.fit(x_train,y_train, epochs=20, batch_size=5, validation_split=0.2,
                 callbacks=[early_stopping, tensorboard])  
print(hist.history) 
#  {'loss': [..2016, ..., 974910], 'mse': [2016., ... , 13.43976]} 요렇게 예쁘게 됩니다요

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])   # epochs를 어떻게 자동으로 인식하는가 싶었는데/ 그냥 1부터 세는거였음
plt.plot(hist.history['val_acc'])

plt.title('loss')
plt.ylabel('evaluate')
plt.xlabel('epoch')
plt.legend(['train loss', 'train_val loss', 'train acc', 'train_val acc']) # 걍 순서대로 들어가네
plt.show()

