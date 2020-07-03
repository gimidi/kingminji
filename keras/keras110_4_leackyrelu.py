import numpy as np
import matplotlib.pyplot as plt
    
def leackyrelu(x) :
    return np.maximum(0.01, x)

x = np.arange(-5,5,0.1)
y = leackyrelu(x)

plt.plot(x,y)
plt.grid()
plt.show()