'''
keras21_ensemble
MSE 는 3.725290298461914e-09
RMSE 는 6.103515625e-05
R2는  0.9999999999153343 튜닝 후 바꿔야함222222222222222222222222
함수형 두개 만들어서 엮을 것임/  sequential은 포함 못해?
'''

#1. 데이터
import numpy as np
x1 = np.array([range(1,101), range(311,411)]).T 
x2 = np.array([range(711,811), range(711,811)]).T

y1 = np.array([range(101,201), range(411,511)]).T 
y2 = np.array([range(501,601), range(711,811)]).T
y3 = np.array([range(411,511), range(611,711)]).T  # 이걸 할당하는 기준을 줘야하는거 아닌가

# W는 1이고 bias는 각각 다른 세트들
# 100행 3열의 데이터셋
# 이제 x1y1 모델1과 x2y2 모델2를 만들어서 합칠 것임

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, random_state=66, test_size=0.2, shuffle=False
)

from sklearn.model_selection import train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, random_state=66, test_size=0.2 , shuffle=False
    )
from sklearn.model_selection import train_test_split
y3_train, y3_test = train_test_split(
    y3, random_state=66, test_size=0.2 , shuffle=False
    )
#2. 모델구성
from keras.models import Sequential, Model 
from keras.layers import Dense, Input 


# 함수형 모델 1
input1 = Input(shape=(2, ), name='input1')
dense1_1 = Dense(16, activation='relu', name='aaa')(input1) # input 레이어를 알려줘야 함/ 노드를 5개로 가지고 input1을 인풋레이어로 받는 댄스층
dense1_2 = Dense(32, activation='relu', name='bbb')(dense1_1) 
dense1_3 = Dense(32, activation='relu', name='ccc')(dense1_2)
#output1 = Dense(3)(dense4) 아웃풋 밑에서 묶어 줌

# 함수형 모델 2
input2 = Input(shape=(2, ), name='input2')
dense2_1 = Dense(16, activation='relu', name='aaa1')(input2) # input 레이어를 알려줘야 함/ 노드를 5개로 가지고 input1을 인풋레이어로 받는 댄스층
dense2_2 = Dense(32, activation='relu', name='bbb1')(dense2_1)
dense2_3 = Dense(32, activation='relu', name='ccc1')(dense2_2)
#output2 = Dense(3)(dense2_4)

# 엮어주쟈~~~ ***************************************************
from keras.layers.merge import concatenate
merge1 = concatenate([dense1_3,dense2_3], name='111') #output 자체를 묶어준다
middle1 = Dense(8, name='222')(merge1)
middle2 = Dense(8, name='333')(middle1)
################ output 모델 구성 #################

# 함수형 모델 1
output1 = Dense(16, name='aaaa')(middle2)
output1_2 = Dense(8, name='bbbb')(output1)
output1_2 = Dense(2, name='cccc')(output1_2)

# 함수형 모델 2
output2 = Dense(5, name='aaaa1')(middle2)
output2_2 = Dense(8, name='bbbb1')(output2)
output2_3 = Dense(2, name='cccc1')(output2_2)

# 함수형 모델 2
output3 = Dense(16, name='aaaa2')(middle2)
output3_2 = Dense(8, name='bbbb2')(output3)
output3_3 = Dense(2, name='cccc2')(output3_2)
#############와 이거 2를 틀리네?ㅋㅋㅋㅋㅋㅋㅋ 뭐지?

model = Model(inputs=[input1, input2],
                outputs=[output1_2,output2_3,output3_3])

model.summary()



#3. 알아듣게 설명한 후 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
######################## 여기서 계속 틀렸음
model.fit([x1_train,x2_train],
            [y1_train,y2_train,y3_train], epochs=250, batch_size=5,
            verbose=2) # verbose 학습중의 정보를 보여주는 버전 고르기 0(아예안나옴),1(다나옴),2(화살표는 안나옴),3(epo만 나옴) 딜레이 시간 아낄려고 -> 1/2 쓰면 될듯
        

#4. 평가
evaluate = model.evaluate([x1_test,x2_test],[y1_test,y2_test,y3_test], batch_size=1)

# a = model.evaluate([x1_test,x2_test],[y1_test,y2_test,y3_train], batch_size=5) # 통상적으로 디폴트값 쓰긴하는데, fit이랑 맞춰줘야 하지 않겠나 싶어서 선생님은 fit값이랑 맞춰준다.
# print('=============================================================')
# print('a 는 ',a)

print('전체loss 는',evaluate[0])
print('output1의loss 는',evaluate[1])
print('optput2의loss 는',evaluate[2])

print('output1의metrics 는',evaluate[3])
print('optput2의metrics 는',evaluate[4])

#5. 예측
y_predict = model.predict([x1_test,x2_test])
print('y_predict 값은', y_predict)

# +추가지표인 MSE와 RMSE 구하기
from sklearn.metrics import mean_squared_error # MSE임
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
#print('전체 MSE 는', mean_squared_error([y1_test,y2_test], y_predict) )
MSE1 = mean_squared_error(y1_test, y_predict[0])
MSE2 = mean_squared_error(y2_test, y_predict[1])
MSE3 = mean_squared_error(y3_test, y_predict[2])
MSE = (MSE1+MSE2+MSE3)/3
# print('output1의 MSE 는', MSE1 )
# print('output2의 MSE 는', MSE2 )
# print('output3의 MSE 는', MSE3 )
print('전체의 MSE 는', MSE )

#print('전체 RMSE 는', RMSE([y1_test,y2_test], y_predict) )
RMSE1 = RMSE(y1_test, y_predict[0])
RMSE2 = RMSE(y2_test, y_predict[1])
RMSE3 = RMSE(y3_test, y_predict[2])
RMSE = (RMSE1+RMSE2+RMSE3)/3
# print('output1의 RMSE 는', RMSE1 )
# print('output2의 RMSE 는', RMSE2 )
# print('output3의 RMSE 는', RMSE3 )
print('전체 RMSE 는', RMSE )


# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y1_test, y_predict[0])
# print('output1의 R2는 ', r2)
r21 = r2_score(y2_test, y_predict[1])
# print('output2의 R2는 ', r21)
r211 = r2_score(y3_test, y_predict[2])
# print('output2의 R2는 ', r211)
print('전체 R2는 ', (r2+r21+r211)/3)

# 아 근데 성능 왜이래....
