import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x) :
    return 1 / (1+ np.exp(-x))  # 이거시 바로 시그모이드 함수~!


x = np.arange(-5,5,0.1)
y = sigmoid(x)

print(x.shape, y.shape)

plt.plot(x,y)
plt.grid()
plt.show()

# 0에서 1로 부드러운 곡선으로 수렴된다
