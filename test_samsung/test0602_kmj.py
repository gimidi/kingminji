import pandas as pd
import numpy as np
삼성전자 = pd.read_csv('삼성전자 주가.csv', encoding='CP949')
하이트 = pd.read_csv('하이트 주가.csv', encoding='CP949')

삼성전자 = 삼성전자.dropna()
하이트 = 하이트.dropna()

삼성전자 = 삼성전자.sort_values(['일자'], ascending=True)
하이트 = 하이트.sort_values(['일자'], ascending=True)

for i in range(len(삼성전자)) :
    삼성전자['시가'][i] = int(삼성전자['시가'][i].replace(',',''))

for j in list(하이트.columns[1:]) :
    for i in range(len(하이트)) :
        하이트[j].iloc[i] = int(하이트[j].iloc[i].replace(',',''))

하이트.loc[509] = ['2020-06-02',39000, 39500, 38500, 38800, 637889]
# 포함하지 말라고 하셔서 밑에 모델 구성 및 예측시 포함하지 않았습니다.

하이트 = 하이트.set_index('일자')
삼성전자 = 삼성전자.set_index('일자')

하이트['삼성전자 시가']=삼성전자['시가']
num_samsung = 삼성전자.values
num_data = 하이트.values

np.save('dataset.npy', arr=num_data)
np.save('samsung.npy', arr=num_samsung)

dataset = np.load('dataset.npy',allow_pickle=True)
samsung = np.load('samsung.npy',allow_pickle=True)

x,y = list(), list()
for i in range(len(dataset)) :
    try:
        x_data = dataset[i:i+5]
        y_data = samsung[i+5]
        x.append(x_data)
        y.append(y_data)
    except:
        break

np.save('x.npy', arr=x)
np.save('y.npy', arr=y)

x = np.load('x.npy',allow_pickle=True)
y = np.load('y.npy',allow_pickle=True)

x1,x2,x3 = list(), list(), list()
for i in range(len(x)) :
    x1.append(x[i][:,0:4])  # 하이트 시가,고가,저가,종가
    x2.append(x[i][:,4])    # 하이트 거래량
    x3.append(x[i][:,5])    # 삼성전가 시가

x1 = np.array(x1)
x2 = np.array(x2)
x3 = np.array(x3)

print(x1.shape)
print(x2.shape)
print(x3.shape)

y = y.reshape(-1)

#전처리
x1 = x1/38750
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x2)
x2 = scaler.transform(x2)

scaler = StandardScaler()
scaler.fit(x3)
x3 = scaler.transform(x3)

# 모델링
from keras.models import Sequential, Model 
from keras.layers import LSTM, Dense, Input 
from keras.callbacks import EarlyStopping, ModelCheckpoint

'''
# 모델(lstm)
x1 = x1.reshape(-1,5,4)
x2 = x2.reshape(-1,5,1)
x3 = x3.reshape(-1,5,1)

from sklearn.model_selection import train_test_split
x1_train, x1_test,x2_train, x2_test,x3_train, x3_test, y_train, y_test = train_test_split(
    x1,x2,x3, y, random_state=66, test_size=0.2 )

# 함수형 모델1
input1 = Input(shape=(5,4))
dense1 = LSTM(64)(input1)
dense1 = Dense(64)(dense1)
dense1 = Dense(64)(dense1)

# 함수형 모델2
input2 = Input(shape=(5,1))
dense2 = LSTM(32)(input2)
dense2 = Dense(64)(dense2)
dense2 = Dense(64)(dense2)

# 함수형 모델3
input3 = Input(shape=(5,1))
dense3 = LSTM(32)(input3)
dense3 = Dense(64)(dense3)
dense3 = Dense(64)(dense3)

from keras.layers.merge import concatenate
merge = concatenate([dense1,dense2,dense3]) 
middle = Dense(160)(merge)
output = Dense(210)(middle)
output = Dense(1)(output)

model = Model(inputs=[input1, input2, input3],outputs=output)
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
modelpath = './model/LSTM/{epoch:03d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1 
                            ,save_best_only=True, save_weights_only=False)

hist = model.fit([x1_train,x2_train,x3_train],
            y_train, epochs=500, batch_size=5, validation_split=0.2,
            verbose=1, callbacks=[checkpoint])

evaluate = model.evaluate([x1_test,x2_test,x3_test],y_test, batch_size=5)
print('loss 는',evaluate[0])

# 모델(dnn)
x1 = x1.reshape(-1,5*4)
x2 = x2.reshape(-1,5)
x3 = x3.reshape(-1,5)

from sklearn.model_selection import train_test_split
x1_train, x1_test,x2_train, x2_test,x3_train, x3_test, y_train, y_test = train_test_split(
    x1,x2,x3, y, random_state=66, test_size=0.2 )

# 함수형 모델1
input1 = Input(shape=(4*5,))
dense1 = Dense(30)(input1)
dense1 = Dense(60)(dense1)

# 함수형 모델2
input2 = Input(shape=(5,))
dense2 = Dense(30)(input2)
dense2 = Dense(60)(dense2)

# 함수형 모델3
input3 = Input(shape=(5,))
dense3 = Dense(600)(input3)
dense3 = Dense(800)(dense3)
dense3 = Dense(1280)(dense3)
dense3 = Dense(600)(dense3)


from keras.layers.merge import concatenate
merge = concatenate([dense1,dense2,dense3]) 
middle = Dense(160)(merge)
output = Dense(210)(middle)
output = Dense(1)(output)

model = Model(inputs=[input1, input2, input3],outputs=output)
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
early_stop = EarlyStopping(monitor='loss', patience=20, mode='auto')
modelpath = './model/DNN/{epoch:03d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1 
                            ,save_best_only=True, save_weights_only=False)
hist = model.fit([x1_train,x2_train,x3_train],
            y_train, epochs=700, batch_size=5, validation_split=0.2,
            verbose=1, callbacks=[early_stop,checkpoint])
evaluate = model.evaluate([x1_test,x2_test,x3_test],y_test, batch_size=5)
print('loss 는',evaluate[0])
'''

from sklearn.model_selection import train_test_split
from keras.models import load_model

# dnn 모델 불러올것임
x1 = x1.reshape(-1,5*4)
x2 = x2.reshape(-1,5)
x3 = x3.reshape(-1,5)

x1_train, x1_test,x2_train, x2_test,x3_train, x3_test, y_train, y_test = train_test_split(
    x1,x2,x3, y, random_state=66, test_size=0.2 )

from sklearn.model_selection import train_test_split
model = load_model('008-608479.3337.hdf5')
evaluate = model.evaluate([x1_test,x2_test,x3_test],y_test, batch_size=5)
print('loss 는',evaluate[0])

predict = model.predict([x1_test[:5],x2_test[:5],x3_test[:5]])
y_test1 = y_test[:5]
x=[]
for i in range(len(y_test1)) : 
    x.append((predict[i]-y_test1[i])[0])
print('예측값은',predict.reshape(-1))
print('정답은',y_test1)
a = [abs(a) for a in x]
print('오차의 총 합은',sum(a))

predict = model.predict([x1_test[-1:],x2_test[-1:],x3_test[-1:]])
print('제가 예측한 0603 시가는 ', np.round(predict,-2),'원 입니다.')


# 0602 데이터를 고려한 답변
last = dataset[-5:]
last1, last2, last3 = list(),list(),list()
for i in range(len(last)) :
    last1.append(last[i][:4])
    last2.append(last[i][4])
    last3.append(last[i][5])
last1 = np.array(last1)
last2 = np.array(last2)
last3 = np.array(last3)
last1 = last1/38750

last1 = last1.reshape(-1,20)
last2 = last2.reshape(-1,5)
last3 = last3.reshape(-1,5)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(last2)
last2 = scaler.transform(last2)

scaler = StandardScaler()
scaler.fit(last3)
last3 = scaler.transform(last3)
predict = model.predict([last1,last2,last3])
print('실제데이터를 넣어서 예측한 0603 시가는 ', np.round(predict,-2),'원 입니다.')
