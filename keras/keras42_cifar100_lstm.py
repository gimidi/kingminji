# 데이터로드
from keras.datasets import cifar100
(x_train, y_train),(x_test,y_test) = cifar100.load_data()  # (50000, 32, 32, 3)

# 전처리
# x는 표준화
x_train = x_train/99
x_test = x_test/99
# y는 원핫 인코딩
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
#----------------------------------------------------------------------------------------------------------------------
# 3차원 모델 (LSTM)
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Dense, Conv2D, Dropout, MaxPooling2D, Flatten
x_train = x_train.reshape(-1,32*32,3)
x_test = x_test.reshape(-1,32*32,3)

input1 = Input(shape=(32*32,3))
layer1 = LSTM(32)(input1)
layer2 = Dense(64)(layer1)
layer3 = Dropout(0.2)(layer2)

layer4 = Dense(128)(layer3)
layer5 = Dense(64)(layer4)
layer6 = Dropout(0.4)(layer5)

output1 = Dense(100, activation='softmax')(layer6)

model = Model(inputs=input1, outputs=output1)
model.summary()
#----------------------------------------------------------------------------------------------------------------------
# 설명과 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stop = EarlyStopping(monitor='loss', patience=10, mode='auto')
# path = './model/{epoch:02d}-{loss:.4f}.hdf5'
# checkpoint = ModelCheckpoint(filepath=path, monitor='val_loss',save_best_only=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train,y_train, epochs=100, batch_size=900, validation_split=0.4)
                # callbacks=[early_stop, checkpoint])

loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)

# 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(12,10))
plt.subplot(2,1,1) # 2행 1열 에 첫번째꺼 그리겠다
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='gray', label='val_loss')
plt.grid()
plt.xlabel('epoch', size=15)
plt.ylabel('loss', size=15)
plt.legend(loc='upper left')

plt.subplot(2,1,2) # 2행 1열 에 두번째꺼 그리겠다
plt.plot(hist.history['acc'], marker='.', c='red', label='acc')  
plt.plot(hist.history['val_acc'], marker='.', c='gray', label='val_acc')
plt.ylabel('acc', size=15)
plt.xlabel('epoch', size=15)
plt.legend(loc='upper left')
# plt.legend(['acc', 'val acc']) # 걍 순서대로 들어가네
plt.show()