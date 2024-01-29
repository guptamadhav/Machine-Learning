import numpy as np
import matplotlib.pyplot as plt
def sigmoid(z):
    g = 1/(1+np.exp(z))
    return g

def error(x, y, w, b):
    m = x.shape[0]
    g_t = np.zeros(m)
    for i in range(m):
        g_t[i] = sigmoid(np.dot(x[i], w) + b)
    return g_t

x_train = np.array([-5,-4,-3,-2,-1,0, 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

w_in = np.array([1])
b_in = 0
g = error(x_train, y_train, w_in, b_in)
# ax = plt.subplot(1,1)
plt.plot(x_train, g, c="b")
plt.show()
