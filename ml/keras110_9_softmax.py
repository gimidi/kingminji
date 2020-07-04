import numpy as np
import matplotlib.pyplot as plt

def softmax(x) :
    return np.exp() / np.sum(np.exp(x))

x = np.arange(1,5)
y = softmax(x)

ratio = y
labels = y

plt.pie(ratio, labels=labels, shadow=True )
plt.show()