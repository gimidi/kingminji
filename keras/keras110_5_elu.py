import numpy as np
import matplotlib.pyplot as plt
    
def elu(x) :
    return np.maximum(5*(np.exp(x)-1), x)

x = np.arange(-5,5,0.1)
y = elu(x)

plt.plot(x,y)
plt.grid()
plt.show()