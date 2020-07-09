from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img

img_dog = load_img('./data/dog_cat/dog.jpg', target_size=(224,224))
img_cat = load_img('./data/dog_cat/cat.jpg', target_size=(224,224))
img_suit = load_img('./data/dog_cat/suit.jpg', target_size=(224,224))
img_yang = load_img('./data/dog_cat/yang.jpg', target_size=(224,224))

plt.imshow(img_yang)
plt.imshow(img_dog)
# plt.show()

from keras.preprocessing.image import img_to_array

arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_suit = img_to_array(img_suit)
arr_yang = img_to_array(img_yang)

print(arr_dog)
print(type(arr_dog)) # <class 'numpy.ndarray'>
print(arr_dog.shape)  # (224, 224, 3)
'''
헐...ㅋㅋㅋㅋ 이렇게 바꿔줌.. 신기하네..
[[[231. 206. 210.]
  [236. 207. 212.]
  [239. 208. 214.]
  ...
'''
# RGB -> BGR (vgg16애들이 이렇게 만들었기 때문에 맞춰주는것)
from keras.applications.vgg16 import preprocess_input
# standard스칼라 형식으로 전처리 한번 해줬다고 생각하면 됨
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_suit = preprocess_input(arr_suit)
arr_yang = preprocess_input(arr_yang)

print(arr_dog.shape) # (224, 224, 3)

# 이미지 데이터를 하나로 합친다

import numpy as np
arr_input = np.stack([arr_dog, arr_cat, arr_suit, arr_yang])

print(arr_input.shape) # (4, 224, 224, 3) 형태 고~대로 합치기만 했다

# 아니 이미지 재미있는데?

# 모델 구성
model = VGG16()
probs = model.predict(arr_input)

print(probs)
print('probs.shape :', probs.shape) # probs.shape : (4, 1000)

# 이미지로 봅시다

from keras.applications.vgg16 import decode_predictions # 예측값을 해석한다

result = decode_predictions(probs)

print('==================================')
print(result[0])
print('==================================')
print(result[1])
print('==================================')
print(result[2])
print('==================================')
print(result[3])


