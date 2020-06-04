from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

#1. 데이터
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]

#2. 모델
# model = SVC()
model = KNeighborsClassifier(n_neighbors=3) # 개체를 하나씩만 연결하겠다
# 최근접 이웃_ 어디까지인지 명시해줌
# 머신러닝이 딥러닝에 밀렸대
# 취업하고나서 그 세세한 모델들을 공부해보기
# 일단 지금은 성능좋은 xgb, rf만..!
# n_neighbors=1일땐 1.0, n_neighbors=3일때 성능 0으로 떨어짐


#3. 실행
model.fit(x_data, y_data)

#4. 평가
x_test = [[0,0],[1,0],[0,1],[1,1]]
y_predict = model.predict(x_test)
print(y_predict)
acc = accuracy_score([0,1,1,0], y_predict)

print('x_test의 예측결과 :',y_predict)
print('acc는 :', acc)