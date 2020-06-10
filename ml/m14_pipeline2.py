# RandomizedSearchCV + PipeLine
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

dataset = load_iris()
x = dataset.data    # -1 , 4
y = dataset.target  # -1 (요소는 0,1,2)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=43)

from sklearn.pipeline import Pipeline , make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# pipe = Pipeline([("scaler",StandardScaler()),('svc',SVC())]) 
pipe = make_pipeline(StandardScaler(),SVC()) # 전처리 방법과 모델

parameters = [
    {'svc__C':[1, 10, 100, 1000], 'svc__kernel':['linear']},    
    {'svc__C':[1, 10, 100], 'svc__kernel':['rbf'], 'svc__gamma':[0.001, 0.0001]}, 
    {'svc__C':[1, 100, 1000], 'svc__kernel':['sigmoid'], 'svc__gamma':[0.001, 0.0001]}  
]

model = RandomizedSearchCV(pipe, parameters, cv=5)
model.fit(x_train, y_train) 
acc = model.score(x_test, y_test)
print("============================")
print("최적의 매개 변수는(estimator):",model.best_estimator_)
print("============================")
print("최적의 매개 변수는(params):",model.best_params_)
print("============================")
print("acc는 ",acc)
