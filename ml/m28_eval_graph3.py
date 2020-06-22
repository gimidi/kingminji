'''
acc :100.00%
'''

from xgboost import XGBRegressor, XGBClassifier, plot_importance 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, r2_score

# 다중분류 모델
dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape) 

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=66)

model = XGBClassifier(objective='multi:softmax', estimators=300, learning_rate=0.1, n_jobs=-1) # 나무의 개수는 결국 epo다
model.fit(x_train, y_train, verbose=True, eval_metric=['merror'], eval_set=[(x_train,y_train),(x_test,y_test)]
        , early_stopping_rounds=20) 

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc :%.2f%%' %(acc*100.0))
print('acc:', acc)
