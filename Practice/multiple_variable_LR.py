import numpy as np
import matplotlib.pyplot as plt
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

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
Y_train = np.array([460, 232, 178])
b_init = 450
w_init = np.array([ 0.89133535, 8.75376741, -23.36032453, -46.42131618])
alpha = 5.0e-7
num = 100000
w_final = np.zeros(X_train.shape[1])
w_final, b_final = gradient_descent(X_train, Y_train, w_init, b_init,alpha, num )
print(w_final, b_final)

for i in range(X_train.shape[0]):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final}, target value: {Y_train[i]}")
m = X_train.shape[0]
yp = np.zeros(m)
for i in range(m):
    yp[i] = np.dot(X_train[i], w_final) + b_final
    # plot predictions and targets versus original features    
fig,ax=plt.subplots(1,4,figsize=(12, 3),sharey=True)
for i in range(len(ax)):
    ax[i].plot(X_train[:,i],yp, label = 'PREDICT')
    ax[i].scatter(X_train[:,i],Y_train,color="orange", label = 'actual')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()
