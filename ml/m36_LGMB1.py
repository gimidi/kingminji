# 이진분류_boston
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
# 이중분류 모델
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=66)

model = LGBMRegressor(n_estimators=200, learning_rate=0.01) # 나무의 개수는 결국 epo다
model.fit(x_train, y_train, verbose=True, eval_metric=['logloss','rmse'], eval_set=[(x_train,y_train),(x_test,y_test)]
        , early_stopping_rounds=20) 

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 score :%.2f%%' %(r2*100.0))

# import pickle
# pickle.dump(model, open("./model/xgb_save/cancer_pickle.dat","wb"))
import joblib 
joblib.dump(model, "./model/xgb_save/cancer.joblib.dat") # 동일하고 wb만 안들어갔음
# modelpath = './model/xgb_save/{rmse:.4f}.model'
# model.save_model(modelpath) # keras가 아주 sklearn 다 배꼈음/ XGB에만 있는듯
print('저장됐다~!')

# model2 = pickle.load(open("./model/xgb_save/cancer_pickle.dat","rb"))
model2 = joblib.load("./model/xgb_save/cancer.joblib.dat")
# model2 = XGBRegressor()
# model2.load_model(modelpath)
print('불러와따!')

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 score :%.2f%%' %(r2*100.0))

thresholds = np.sort(model.feature_importances_)
print(thresholds)

for thresh in thresholds : # 컬럼수만큼 돈다! 빙글빙글
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    selection_x_train = selection.transform(x_train)
    # print(selection_x_train.shape) # 칼럼이 하나씩 줄고 있는걸 알 수 있음 (가장 중요 x를 하나씩 지우고 있음)
    
    selection_model = LGBMRegressor()
    selection_model.fit(selection_x_train, y_train)

    select_x_test = selection.transform(x_test)
    x_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, x_pred)
    print('R2는',score)
    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, selection_x_train.shape[1], score*100.0))


