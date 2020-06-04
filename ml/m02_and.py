from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#1. 데이터
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,0,0,1]

#2. 모델
model = LinearSVC()

#3. 실행
model.fit(x_data, y_data)

#4. 평가
x_test = [[0,0],[1,0],[0,1],[1,1]]
y_predict = model.predict(x_test)
print(y_predict)
acc = accuracy_score([0,0,0,1], y_predict)
# 얘가 evaluate 대신 sklearn에 들어있는 애
아이구 예뻐라
print('x_test의 예측결과 :',y_predict)
print('acc는 :', acc)