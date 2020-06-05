import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators  # sklearn 0.20.1에서만 돌아감요
import warnings
warnings.filterwarnings('ignore')

iris = pd.read_csv('./data/csv/iris.csv',header=0)
x = iris.iloc[:,:4]
y = iris.iloc[:,4]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=66)

allAlgorithms = all_estimators(type_filter='classifier')

for (name, algorithm) in allAlgorithms :
    model = algorithm()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(name, '의 정답률 = ', accuracy_score(y_test, y_pred))

import sklearn
print('sklearn의 버전은 ' , sklearn.__version__)


