# iris를 케라스 파이프라인 구성
# 당연히 RandomizedSearchCV 구성
# keras 98 참조할것
# 97번을 RandomizedSearchCV로 변경

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, Dense, MaxPool2D

dataset = load_iris()
x = dataset.data   
y = dataset.target 
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=43)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(x_train.shape)
print(x_test.shape)  
print(y_train.shape) 
print(y_test.shape)

# def build_model(optimizer='adam', dropout=0.5) :
#     input = Input(shape=(4,))
#     x = Dense(64, activation='relu')(input)
#     x = Dropout(0.5)(x)
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     output = Dense(3, activation='softmax')(x)
#     model = Model(inputs=input, outputs=output)
#     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
#     return model
    
def build_model(optimizer='adam', dropout=0.2, node=60) :  # optimizer and dropout and .... is not a legal parameter
    input = Input(shape=(4,))
    x = Dense(64, activation='relu')(input)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(3, activation='softmax')(x)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    return model

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor 
model = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.pipeline import Pipeline , make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
pipe = Pipeline([("scaler",StandardScaler()),('model', model)]) 
# pipe = make_pipeline(StandardScaler(),model) # model말고 KerasClassifier을 그대로 넣어야함요!

def create_hyperparameters() :
    batches = [50,500]  
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.3,0.5, 0.7]
    epochs = [100,200]
    node = [40,80]  
    return {'model__batch_size': batches, "model__optimizer": optimizers, "model__dropout":dropout,
            'model__epochs':epochs, 'model__node':node}
hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(pipe, hyperparameters, cv=3) 

search.fit(x_train, y_train)
acc = search.score(x_test, y_test)
print("============================")
# print('best_estimator_은', search.best_estimator_)
# print("============================")
print('best_params_은', search.best_params_)
print("============================")
print('acc은 ',acc)

'''
acc 1.0
best_params_은 
{'model__optimizer': 'adadelta', 'model__node': 80, 'model__epochs': 200, 
'model__dropout': 0.5, 'model__batch_size': 500}
'''
'''
acc 1.0
best_params_은 {'model__optimizer': 'rmsprop', 'model__node': 80, 'model__epochs': 200,
 'model__dropout': 0.7, 'model__batch_size': 50}
'''