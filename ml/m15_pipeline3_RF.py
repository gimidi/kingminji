# RandomizedSearchCV + PipeLine
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

dataset = load_iris()
x = dataset.data   
y = dataset.target 
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=43)

from sklearn.pipeline import Pipeline , make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pipe = Pipeline([("scaler",StandardScaler()),('rf',RandomForestClassifier())]) 
# pipe = make_pipeline(StandardScaler(),RandomForestClassifier()) # 전처리 방법과 모델

parameters = [
    { 'rf__max_depth':[100, 200, 300],'rf__n_estimators':[10, 50, 150]},
    {'rf__max_depth':[200, 300], 'rf__max_leaf_nodes':[10,50,150]},
    {'rf__n_estimators':[10, 50, 150,250], 'rf__max_leaf_nodes':[1,2,3,4,5,10,50],  'rf__max_depth':[200, 300, 500]}
]

model = RandomizedSearchCV(pipe, parameters, cv=5)
model.fit(x_train, y_train) 
acc = model.score(x_test, y_test)

print("============================")
print("최적의 매개 변수는(estimator):", model.best_estimator_) # 전체 params 다 나옴
print("============================")
print("최적의 매개 변수는(params):", model.best_params_)
print("============================")
print("acc는 ",acc)

'''
============================
최적의 매개 변수는(params): {'rf__n_estimators': 150, 'rf__max_leaf_nodes': 4, 'rf__max_depth': 200}
============================
'''