import numpy as np
import pandas as pd

datasets = pd.read_csv('./data/csv/iris.csv', index_col=None, header=0, sep=',')
# header=0 이게 header 표시하겠다였어...? 인덱스랑 헤드 다 나오는 이유는 멉니까?
# 아 여기꺼 안쓰고 자동인덱스 쓰겠다 이거래.....ㅎ
# 첫번째 라인을 header로 하겠다. -> 1행부터가 실 데이터임을 알려주는것임
# header=None 으로 하면 첫번째행부터 실 데이터로 보고 새로운 칼럼명 자동으로 붙여줌

print(datasets)
print(datasets.head())
print(datasets.tail())

print(datasets.values)  # pandas -> numpy 형식으로 바꿔줌
aaa = datasets.values
print(type(aaa))     # <class 'numpy.ndarray'>

# 넘파이로 저장하시오

# np.save('./data/iris.npy', arr=aaa)
