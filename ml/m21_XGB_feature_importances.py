from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=44, test_size=0.2
)

model = XGBClassifier(max_depth=3, min_samples_leaf=1) # depth 깊으면 과대적합됨
# max_depth : 기본값 써라!
# n_estimators : 클수록 좋다!, 단점은 메모리 짱 차지, 기본값 100
# n_jobs=-1 : gpu 병렬처리

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

print(model.feature_importances_)

import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_cancer(model) :
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("feature importance")
    plt.ylabel("features")
    plt.ylim(-1, n_features)
plot_feature_importances_cancer(model)
plt.show()
