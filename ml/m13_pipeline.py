import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

dataset = load_iris()

x = dataset.data    # -1 , 4
y = dataset.target  # -1 (요소는 0,1,2)

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=13)

# 전처리 친구가 파이프라인 -> 전처리 한번에 적용시켜준다.

# model = SVC()
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# pipe = make_pipeline(StandardScaler(),SVC()) # 전처리 방법과 모델
pipe = Pipeline([("scaler",StandardScaler()),("svm",SVC())]) # 전처리 방법과 모델
# 2차원이라서 바로 넘어가는건가...

pipe.fit(x_train, y_train) 
# pipe는 모델임..

print("acc:", pipe.score(x_test,y_test))


