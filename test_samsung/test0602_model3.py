# LSTM 2개 구현

import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def split_x(seq, size) :
    aaa = []
    for i in range(len(seq) - size + 1) :
        subset = seq[i:(i+size)]
        aaa.append([j for j in subset])
    return np.array(aaa)

# 데이터
samsung = np.load('./data/samsung.npy', allow_pickle=True)
hite = np.load('./data/hite.npy', allow_pickle=True)
# samsung = samsung.reshape(-1)
size = 6
samsung = split_x(samsung, size)

x_sam = samsung[:, 0:5]
y_sam = samsung[:,5]
x_hit = hite[5:510,:]
print(x_sam.shape)  # (504, 5, 1)
print(x_hit.shape)  # (504, 5)
print(y_sam.shape)  # (504, 1)
y_sam = y_sam.reshape(-1)
x_hit = x_hit.reshape(-1,5,1)


# LSTM
input1 = Input(shape=(5,1))
x1 = LSTM(10)(input1)
x1 = Dense(10)(x1)

input2 = Input(shape=(5,1))
x2 = LSTM(10)(input2)
x2 = Dense(10)(x2)

merge = concatenate([x1, x2])
output = Dense(1)(merge)

model = Model(inputs=[input1, input2], outputs= output)
model.summary()

model.compile(optimizer='adam', loss='mse')
model.fit([x_sam,x_hit], y_sam, epochs=5)
