'''
acc :96.49%
'''

from xgboost import XGBRegressor, XGBClassifier, plot_importance # 중요한걸 그리겠네
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, r2_score

# 이중분류 모델
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape) 

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=66)

# XGB에만 있는 애들인거 같아
model = XGBClassifier(n_estimators=300, learning_rate=0.01) # 나무의 개수는 결국 epo다
model.fit(x_train, y_train, verbose=True, eval_metric=['error','auc'], eval_set=[(x_train,y_train),(x_test,y_test)]
        , early_stopping_rounds=50) # topping. Best iteration:  validation_0-rmse:0.57088       validation_1-rmse:2.36899
# validation이 기준이네? 더 떨어지는게 없는게 20번째인걸 과적합으로 봤을때 그게 시작되는 시점을 잡아준다
# verbose? 뭔가 보여주는거 같네?
# metric? 성능평가 지표인거 같네..? 옵션(rmse, mae, logloss, error, auc) # error가 acc, auc는 acc 칭구
# validation_0은 train 값 / validation_1은 test 값임
result = model.evals_result_
# print("eval's results : ",result) 

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc :%.2f%%' %(acc*100.0))
print('acc:', acc)

import matplotlib.pyplot as plt
epochs = len(result['validation_0']['error'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['error'], label='Train')
ax.plot(x_axis, result['validation_1']['error'], label='Test')
ax.legend()
plt.ylabel('error')
plt.title('XGBoost error')
plt.show()

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['auc'], label='Train')
ax.plot(x_axis, result['validation_1']['auc'], label='Test')
ax.legend()
plt.ylabel('auc')
plt.title('XGBoost auc')
plt.show()

# parameters = [
#     {"n_estimators":[100,200,300,50,80,90,120,150], "learning_rate":[0.1, 0.3, 0.5, 0.01, 0.005, 0.005, 0.09,0.001],
#     "max_depth":[3,5,7,20,50,3,4,5,100,150], "colsample_bylevel":[0.6,0.8,1,0.6,0.7,0.8,0.9,1]},
#     # {"n_estimators":[50,80,90,100,120,150], "learning_rate":[0.1, 0.005, 0.09,0.001],
#     # "max_depth":[3,4,5,100,150],"colsample_bytree":[0.6,0.7,0.8,0.9,1] }
# ]

# model = GridSearchCV(XGBRegressor(),parameters,cv=5,n_jobs=-1)
# model.fit(x_train, y_train)
# score = model.score(x_test, y_test)
# print('점수 : ', score)
# # print(model.feature_importances_)
# print('============================================')
# print('best_estimator_',model.best_estimator_)
# print('============================================')
# print('best_params_',model.best_params_)