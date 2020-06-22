from xgboost import XGBRegressor, XGBClassifier, plot_importance 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel
import numpy as np

# 다중분류 모델
dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape) 

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=66)

model = XGBClassifier(objective='multi:softmax', estimators=300, learning_rate=0.01, n_jobs=-1) # 나무의 개수는 결국 epo다
model.fit(x_train, y_train, verbose=True, eval_metric=['merror'], eval_set=[(x_train,y_train),(x_test,y_test)]
        , early_stopping_rounds=20) 

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc :%.2f%%' %(acc*100.0))
print('acc:', acc)

modelpath = './model/xgb_save/auc.dat'
import pickle
pickle.dump(model, open(modelpath,"wb"))
# import joblib 
# joblib.dump(model, modelpath) 
# model.save_model(modelpath) # keras가 아주 sklearn 다 배꼈음
print('저장됐다~!')

model2 = pickle.load(open(modelpath,"rb"))
# model2 = joblib.load(modelpath)
# model2 = XGBRegressor()
# model2.load_model(modelpath)
print('불러와따!')

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc :%.2f%%' %(acc*100.0))
print('acc:', acc)

thresholds = np.sort(model.feature_importances_)

for thresh in thresholds : # 컬럼수만큼 돈다! 빙글빙글
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    selection_x_train = selection.transform(x_train)
    # print(selection_x_train.shape) # 칼럼이 하나씩 줄고 있는걸 알 수 있음 (가장 중요 x를 하나씩 지우고 있음)
    
    selection_model = XGBRegressor()
    selection_model.fit(selection_x_train, y_train)

    select_x_test = selection.transform(x_test)
    x_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, x_pred)
    print('R2는',score)
    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, selection_x_train.shape[1], score*100.0))