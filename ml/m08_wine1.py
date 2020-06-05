import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

wine = pd.read_csv('winequality-white.csv',sep=';')

x = wine.iloc[:,0:-1]
y = wine.iloc[:,-1]

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(
    x, y, random_state=66, test_size=0.2 )

model = RandomForestClassifier()
model.fit(x_train,y_train)
score = model.score(x_test,y_test) # 회귀던 분류던 사용할 수 있음
print('RandomForestClassifier score(acc)는',score)

# y_predict = model.predict(x_test)
# acc = accuracy_score(y_predict,y_test)
# print('RandomForestClassifier acc는',acc)
