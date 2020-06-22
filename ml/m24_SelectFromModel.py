from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score

dataset = load_boston()
x = dataset.data
y = dataset.target

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=66)

model = XGBRegressor()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('디폴트 score은',score)

# 우리가 지금 할 것은 피처 엔지니어링이다.

thresholds = np.sort(model.feature_importances_)

print(thresholds)

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

# 그리드 서치 엮어라

# 데이콘 적용해라. 71개 칼럼 -> 성적 메일로 제출하기

# 메일 제목 : 말똥이 10등

# 24_2, 3번 파일 만들어라

# 일요일 23시 59분