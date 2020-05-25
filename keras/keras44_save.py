'''

'''

import numpy as np
from keras.models import Sequential 
from keras.layers import LSTM, Dense 

model = Sequential()
model.add(LSTM(320, input_shape=(4,1))) # input을 넣는거야 무조권 ^_^ 
model.add(Dense(32))
model.add(Dense(640))
model.add(Dense(640))
model.add(Dense(16))  
model.add(Dense(10)) 

model.summary()

model.save("./model/save44.h5")
print('저장 잘 됐다 !')