# 과적합 방지
# 1. 훈련데이터를 늘린다
# 2. 피처수를 줄인다.
# 3. regularization


# 생체광학 다음주에 끝나니까 그거 들어가라

from xgboost import XGBRegressor, plot_importance # 중요한걸 그리겠네
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

x,y=load_boston(return_X_y=True)
print(x.shape) # (506, 13)
print(y.shape) # (506,)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=66)

n_estimators = 150  # 트리 개수
learning_rate = 0.059  # 학습률 
colsample_bytree = 1 # 90% 
colsample_bylevel = 0.6 # 어느정도 활용하겠는가? max는 1 / 0.6~0.9로 쓰기

n_jobs = -1  # 딥러닝이 아닐경우는 -1 (딥러닝이면 터짐요..)
max_depth = 5 # 큰 영향을 주지 않음

model = XGBRegressor(
max_depth=max_depth, 
learning_rate=learning_rate, # 가장 중요한 것
n_estimators=n_estimators, n_jobs=n_jobs,
# colsample_bylevel=colsample_bylevel,
colsample_bytree=colsample_bytree
)

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('점수 : ', score)
# print(model.feature_importances_)


# print(model.best_iteration)
# print(model.best_ntree_limit)

# plot_importance(model)
# plt.show()