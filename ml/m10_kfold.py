import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators  # sklearn 0.20.1에서만 돌아감요
import warnings
warnings.filterwarnings('ignore')

iris = pd.read_csv('./data/csv/iris.csv',header=0)
x = iris.iloc[:,:4]
y = iris.iloc[:,4]

# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=66)

Kfold = KFold(n_splits=5, shuffle=True) # 5등분으로 하겠다 -> 5번 돈다 (=데이터의 모든 부분이 test영역이 된다)
# 100하면 100등분하고 100번 돈다 (걍 어쨌든 모든 데이터가 test에 한번은 들어간다는 뜻임)
# 26모델 * 5번씩 돌리니까 130번 돌아간다. 아 ml는 epo가 없어서 kflod하는구나

allAlgorithms = all_estimators(type_filter='classifier')

for (name, algorithm) in allAlgorithms :
    model = algorithm()
    scores = cross_val_score(model, x, y, cv=Kfold) # fit이랑 score(acc) 한꺼번에 함
    # 평균치를 해서 가장 성능 좋은 모델을 찾는게 좋겠다.
    print(name, '의 정답률은 = ')
    print(sum(scores)/len(scores))
    # model.fit(x, y)
    # y_pred = model.predict(x)
    # print(name, '의 정답률 = ', accuracy_score(y_test, y_pred))

import sklearn
print('sklearn의 버전은 ' , sklearn.__version__)


