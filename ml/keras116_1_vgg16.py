# VGG-16 모델은 Image Net Challenge에서 Top-5 테스트 정확도를 92.7% 달성

from keras.applications import VGG16, VGG19
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation
from keras.optimizers import Adam

vgg16 = VGG16()
vgg19 = VGG19()

model = Sequential()
model.add(vgg16)
# model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# model = Sequential()
# model.add(vgg19)
# # model.add(Flatten())
# model.add(Dense(256))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(10, activation='softmax'))

# model.summary()
