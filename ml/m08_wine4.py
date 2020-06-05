import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 와인 데이터 읽기
wine = pd.read_csv('winequality-white.csv',sep=';')

y = wine['quality']
x = wine.drop('quality', axis=1)

print(x.shape)
print(y.shape)

# 3~9를 0,1,2로 축소할것임

newlist = []
for i in list(y) :
    if i <=4 :
        newlist +=[0]
    elif i <=7 :
        newlist +=[1]
    else:
        newlist +=[2]

y = np.array(newlist)
print(x.shape)
print(y.shape)
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(
    x, y, random_state=66, test_size=0.2 )

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train,y_train)
acc = model.score(x_test, y_test)
y_pred = model.predict(x_test)
print('정답률은', accuracy_score(y_pred,y_test))
print('scroe(acc)는', acc)