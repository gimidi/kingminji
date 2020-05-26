from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten # 2차원/ 이미지는 2차원임
# 이미지를 레이어마다 짜르는거니까 layer에 들어있다

model = Sequential()
# model.add(Conv2D(10, (2,2), 
#             input_shape=(5,5,1) )) # 이미지를 가로x세로 2,2로 짜르겠다
#             # ( , ,3) 칼라 ( , ,1) 흑백 / (10,10,1) 가로,세로,흑백 이게 10000장이면 
#             # 10000, 10, 10, 1  이렇게 되는것임
#             # 가로 세로 정보는 필요할거 같긴한데...
#             # 흠,, LSTM도 3차원인데 사실상 inputdim=1로 돌아갔단 말이지 / CNN은 어떨까?

            # 머신러닝 라이브러리는 다음에 봤을때 사용할 수 있을정도로 이해하기만 하면 됨 <- 그게 포인트임
model.add(Conv2D(10, (2,2), input_shape=(10,10,1))) # (None, 9 , 9, 10)
model.add(Conv2D(7, (3,3))) # (7,7,7))
model.add(Conv2D(5, (2,2), padding='same')) # (7,7,5) 와 패딩 한쪽만 해..?
model.add(Conv2D(5, (2,2))) # (6,6,5))
# model.add(Conv2D(5, (2,2), strides=2))  # (3,3,5) 6/2=3
# model.add(Conv2D(5, (2,2), strides=2, padding='same'))  # (3,3,5) 6/2=3
model.add(MaxPooling2D(pool_size=2)) # (None, 3, 3, 5)
# 4차원을 2차원으로 쫙 펴준다
model.add(Flatten())
model.add(Dense(1))
model.summary()