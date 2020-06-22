from xgboost import XGBRegressor,XGBClassifier, plot_importance # 중요한걸 그리겠네
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, r2_score

# 다중분류 모델

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

print(x.shape) # (150, 4)
print(y.shape) # (150,)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=66)

# XGB에만 있는 애들인거 같아
# model = XGBRegressor(n_estimators=3, learning_rate=0.1) # 나무의 개수는 결국 epo다
model = XGBClassifier(n_estimators=300, learning_rate=0.01) # 나무의 개수는 결국 epo다
model.fit(x_train, y_train, verbose=True, eval_metric=['error','auc'], eval_set=[(x_train,y_train),(x_test,y_test)]
        , early_stopping_rounds=20) 
# result = model.evals_result_
 
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc :%.2f%%' %(acc*100.0))
print('acc:', acc)

# import pickle
# pickle.dump(model, open("./model/xgb_save/cancer_pickle.dat","wb"))
# import joblib 
# joblib.dump(model, "./model/xgb_save/cancer.joblib.dat") # 동일하고 wb만 안들어갔음
model.save_model("./model/xgb_save/cancer.xgb.model") # keras가 아주 sklearn 다 배꼈음
print('저장됐다~!')

# model2 = pickle.load(open("./model/xgb_save/cancer_pickle.dat","rb"))
# model2 = joblib.load("./model/xgb_save/cancer.joblib.dat")
model2 = XGBClassifier()
model2.load_model("./model/xgb_save/cancer.xgb.model")
print('불러와따!')

y_pred = model2.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc :%.2f%%' %(acc*100.0))
print('acc:', acc)

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