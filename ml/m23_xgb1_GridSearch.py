from xgboost import XGBRegressor, plot_importance # 중요한걸 그리겠네
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

# 다중분류 모델

dataset = load_boston()
x = dataset.data
y = dataset.target

print(x.shape) # (150, 4)
print(y.shape) # (150,)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=66)

parameters = [
    {"n_estimators":[100,200,300,50,80,90,120,150], "learning_rate":[0.1, 0.3, 0.5, 0.01, 0.005, 0.005, 0.09,0.001],
    "max_depth":[3,5,7,20,50,3,4,5,100,150], "colsample_bylevel":[0.6,0.8,1,0.6,0.7,0.8,0.9,1]},
    # {"n_estimators":[50,80,90,100,120,150], "learning_rate":[0.1, 0.005, 0.09,0.001],
    # "max_depth":[3,4,5,100,150],"colsample_bytree":[0.6,0.7,0.8,0.9,1] }
]

model = GridSearchCV(XGBRegressor(),parameters,cv=5,n_jobs=-1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('점수 : ', score)
# print(model.feature_importances_)
print('============================================')
print('best_estimator_',model.best_estimator_)
print('============================================')
print('best_params_',model.best_params_)