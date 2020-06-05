# 보스턴 모델링하시오

from sklearn.datasets import load_boston
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils.testing import all_estimators  # sklearn 0.20.1에서만 돌아감요
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv('./data/csv/boston_house_prices.csv', header=1)
x = dataset.iloc[:,0:-1]
y = dataset.iloc[:,-1]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x) #  전처리를 실행하다 
x = scaler.transform(x)

# from sklearn.preprocessing import MinMaxScaler # 별 차이 없음..!
# scaler = MinMaxScaler()
# scaler.fit(x) #  전처리를 실행하다 
# x = scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.2)

allAlgorithms = all_estimators(type_filter='regressor')

for (name, algorithm) in allAlgorithms :
    model = algorithm()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # print(name, '의 score(r2) = ', model.score(x_test, y_test))
    print(name, '의 r2 = ', r2_score(y_pred, y_test))

import sklearn
print('sklearn의 버전은 ' , sklearn.__version__)
