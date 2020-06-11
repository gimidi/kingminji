# 97번을 RandomizedSearchCV로 변경

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, Dense, MaxPool2D
import numpy as np

# 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape) # (60000, 28, 28)
print(x_test.shape)  # (10000, 28, 28)
print(y_train.shape) # (60000,)
print(y_test.shape)  # (10000,)

# x_train = x_train.reshape(-1,28,28,1)/255 
# x_test = x_test.reshape(-1,28,28,1)/255
x_train = x_train.reshape(-1,28*28)/255 
x_test = x_test.reshape(-1,28*28)/255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(x_train.shape)  # (60000, 784)
print(x_test.shape)   # (10000, 784)
print(y_train.shape)  # (60000, 10)
print(y_test.shape)   # (10000, 10)

# 모델
# 모델자체를 함수로 만들거다

def build_model(drop=0.5, optimizer='adam') :
    input = Input(shape=(28*28,))
    x = Dense(512, activation='relu')(input)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(drop)(x)
    output = Dense(10, activation='softmax')(x)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    return model
    
def create_hyperparameters() :
    batches = [10,20,30,40,50]  # 이거말고도 epo, node, activation 다 넣을 수 잇음...
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1,0.5,5)  
    # 현재의 경우의 수는 5*3*5 * (cv 3)
    return {'batch_size': batches, "optimizer": optimizers, "drop": dropout}

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor # 분류, 회귀모델에 대한 사이킷런 형식을 랩핑하겠다
model = KerasClassifier(build_fn=build_model, verbose=1)
hyperparameters = create_hyperparameters()

# 드디어 그리드서치

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model, hyperparameters, cv=3) # train에 2/3가 들어가고 있음 () # n_jobs가 cpu 갯수라는데 -1해도 터지고..(걍 뻑나감) 7해도 터지고..

# 그리드서치 자체를 fit 때려버리네?
search.fit(x_train, y_train)
acc = search.score(x_test, y_test)

print('best_params_은', search.best_params_)
print('sore은 ',acc)

