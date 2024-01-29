import numpy as np
import matplotlib.pyplot as plt
import copy, math

def sigmoid(z):
    g = 1/(1+np.exp(-z))
    return g

def compute_cost_logistic(x, y, w, b):
    m,n = x.shape
    cost = 0
    for i in range(m):
        z = np.dot(x[i], w)+b
        f_wb = sigmoid(z)
        error = -y[i]*np.log(f_wb)-(1-y[i])*(np.log(1-f_wb))
        cost += error
    total_cost = cost/m
    return total_cost

def compute_gradient_logistic(x,y,w,b):
    m,n =  x.shape
    df_dw_i = np.zeros((n,))
    df_db_i = 0.
    for i in range(m):
        z = np.dot(w,x[i]) + b
        f_wb = sigmoid(z)
        err = f_wb - y[i]
        for j in range(n):
            df_dw_i[j] += err*x[i, j]
        df_db_i += err
    df_dw = df_dw_i/m
    df_db = df_db_i/m
    return df_dw, df_db

def gradient_descent(x,y,w_in,b_in,alpha,num):
    J_history = []
    # in order to prevent orignal values of parameter from being modified
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num):
        df_dw, df_db = compute_gradient_logistic(x,y,w,b)
        w = w - alpha*df_dw
        b = b - alpha*df_db
        if i<1000000:
            J_history.append(compute_cost_logistic(x,y,w,b))
        if i%(math.ceil(num/10)) == 0: # print at every 10% of the value
            print(f"{i}) cost: {compute_cost_logistic(x,y,w,b)}, w: {w}, b:{b}")
    return w,b,J_history

def plot_data(X, y, ax, pos_label="y=1", neg_label="y=0", s=80, loc='best' ):
    """ plots logistic data with two axis """
    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0
    pos = pos.reshape(-1,)  #work with 1D or 1D y vectors
    neg = neg.reshape(-1,)

    # Plot examples
    ax.scatter(X[pos, 0], X[pos, 1], marker='x', s=s, c = 'red', label=pos_label)
    ax.scatter(X[neg, 0], X[neg, 1], marker='o', s=s, label=neg_label, facecolors='none',  lw=3)
    ax.legend(loc=loc)

    ax.figure.canvas.toolbar_visible = False
    ax.figure.canvas.header_visible = False
    ax.figure.canvas.footer_visible = False

X_tmp = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_tmp = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.array([2.,3.])
b_tmp = 1.
dj_dw_tmp, dj_db_tmp = compute_gradient_logistic(X_tmp, y_tmp, w_tmp, b_tmp)
print(f"dj_db: {dj_db_tmp}" )
print(f"dj_dw: {dj_dw_tmp.tolist()}" )
w_tmp  = np.zeros_like(X_tmp[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out, _ = gradient_descent(X_tmp, y_tmp, w_tmp, b_tmp, alph, iters) 
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")
# # plotting decision boundary : decision boundary is basically separating between 0 and 1 values which happens when 
# # z = 0 , (therefore w_0*x_0 + w_1*x_1 + b = 0) as at z=0 the sigmoid function gives 0.5 which separates both values
# x0 = np.arange(0,6) #choose values between 0 and 6
# #for different values of b
# x1 = 3 - x0
# x1_other = 4 - x0
# fig,ax = plt.subplots(1, 1, figsize=(4,4))
# # Plot the decision boundary
# ax.plot(x0,x1, label="$b$=-3")
# ax.plot(x0,x1_other, label="$b$=-4")
# ax.axis([0, 4, 0, 4])
# plt.scatter(X_train, y_train)
# plt.show()
fig,ax = plt.subplots(1,1,figsize=(5,4))
# plot the probability 
# plt_prob(ax, w_out, b_out)

# Plot the original data
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')   
ax.axis([0, 4, 0, 3.5])
plot_data(X_tmp,y_tmp,ax)

# Plot the decision boundary
x0 = -b_out/w_out[0]
x1 = -b_out/w_out[1]
ax.plot([0,x0],[x1,0],  lw=1)
plt.show()