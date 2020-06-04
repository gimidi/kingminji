import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,10,0.1) # 0부터 10까지 0.1씩 증가시킨다
y = np.sin(x)

plt.plot(x,y)

plt.show()