import numpy as np
from matplotlib import pyplot as plt
#feature engineering and polynomial regression
def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = np.dot(w, x[i]) + b
        cost += (f_wb - y[i])**2

    total_cost = 1/(2*m) * cost

    return total_cost

def compute_gradient(x, y, w, b):
    m,n = x.shape
    df_dw_i = np.zeros(n)
    df_db_i = 0
    for i in range(m):
        err = (np.dot(w, x[i]) + b) - y[i]
        for j in range(n): 
            # basically same as in single  variable. here the difference is that we need to add to each element
            # of w seperately in order to maintain multiple variables.First for i=0, we add to all w multiplied
            # by specific x[i, j] and then added consequtively for other values of i to different elements of w
            df_dw_i[j] += err*x[i, j]
        df_db_i += err
    df_dw = df_dw_i/m
    df_db = df_db_i/m

    return df_dw, df_db

def gradient_descent(x, y, w, b, alpha, num):
    m = x.shape[0]
    for i in range(num):
        df_dw, df_db = compute_gradient(x,y,w,b)
        w = w - alpha*df_dw
        b = b - alpha*df_db

        if i%100 == 0 :
            print(f"{i}) Cost = {compute_cost(x,y,w,b)}, w = {w}, b = {b}")
    return w, b

x = np.arange(0, 20, 1)
y = 1 + x**2
# Engineer features 
X = x**2      #<-- added engineered feature

X = X.reshape(-1,1) # reshape(-1,1) converts 1-d array into 2-d such that all elements as columns will now be rows 
                    # with 1 element each
m,n = X.shape
w = np.zeros(n)
b = 0
w_final, b_final = gradient_descent(X,y,w,0,alpha=1e-6, num=10000)
plt.scatter(x,y, marker="x", label = "actual value")
plt.plot(x,(np.dot(X, w_final)+b_final), label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
plt.show()
# x = np.arange(0,20,1)
# y = np.cos(x/2)
# print(x)

# X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
# what np.c_ does is it concatenate elements of various same columns of an array into one row
# print(X)