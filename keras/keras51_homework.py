<<<<<<< HEAD
# 2번의 첫번째 답

import numpy as np
y = np.array([1,2,3,4,5,1,2,3,4,5])

from keras.utils import np_utils
y = np_utils.to_categorical(y)
y = y - 1 # 넘파이는 이럼 되지 않어?! 하면 됨 ㅎㅎ
print(y)

# x = [1,2,3]
# x = x - 1 # 리스트는 안됨요 / ex뭐시기 함수는 있었던듯!
# print(x) 


# 2번의 두번째 답
from sklearn.preprocessing import OneHotEncoder
y = y.reshape(10,1)
aaa = OneHotEncoder()
aaa.fit(y)
y = aaa.transform(y).toarray()
# 원핫 인코딩은 2차원으로 넣어줘야 한다?

=======
# 2번의 첫번째 답

import numpy as np
y = np.array([1,2,3,4,5,1,2,3,4,5])

from keras.utils import np_utils
y = np_utils.to_categorical(y)
y = y - 1 # 넘파이는 이럼 되지 않어?! 하면 됨 ㅎㅎ
print(y)

# x = [1,2,3]
# x = x - 1 # 리스트는 안됨요 / ex뭐시기 함수는 있었던듯!
# print(x) 


# 2번의 두번째 답
from sklearn.preprocessing import OneHotEncoder
y = y.reshape(10,1)
aaa = OneHotEncoder()
aaa.fit(y)
y = aaa.transform(y).toarray()
# 원핫 인코딩은 2차원으로 넣어줘야 한다?

>>>>>>> fa2732da2f961158aee21e0eecf6dd0dd4f77931
