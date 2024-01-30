import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    g = 1/(1+np.exp(-z))
    return g

def compute_cost_linear_reg(x, y, w, b, lambda_):
    m,n = x.shape
    cost = 0
    for i in range(m):
        f_wb = np.dot(w,x[i]) + b
        err = (f_wb - y[i])**2
        cost += err
    cost = cost/(2*m)
    #regularisation cost
    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)
    reg_cost = reg_cost*(lambda_/(2*m))

    total_cost = cost + reg_cost

    return total_cost

def compute_cost_logistic_reg(x, y, w, b, lambda_):
    m,n = x.shape
    cost = 0
    for i in range(m):
        z = np.dot(w,x[i])+b
        f_wb = sigmoid(z)
        loss = -y[i]*np.log(f_wb)-(1-y[i])*np.log(1-f_wb)
        cost += loss
    cost = cost/m
    #regularisation cost
    reg_cost = 0
    for j in range(n):
        reg_cost += w[j]**2
    reg_cost = lambda_/(2*m)*reg_cost
    
    total_cost = cost+reg_cost
    return total_cost

def compute_gradient_linear_reg(x,y,w,b,lambda_):
    m,n = x.shape
    df_dw_i  = np.zeros(n)
    df_db_i = 0
    for i in range(m):
        f_wb = np.dot(w,x[i])+b
        err = f_wb-y[i]
        for j in range(n):
            df_dw_i[j] += err*x[i,j]
        df_db_i += err
    df_dw = df_dw_i/m + lambda_/m*w # or could have looped in w and individually add w[j]*lambda_/m to each of its respective value
    df_db = df_db_i/m

    return df_dw, df_db

def compute_gradient_logistic_reg(x, y, w, b, lambda_):
    m,n = x.shape
    df_dw = np.zeros((n,))
    df_db = 0.
    for i in range(m):
        z = np.dot(w, x[i]) + b
        f_wb = sigmoid(z)
        loss = f_wb - y[i] #y predicted minus actual value of y[i]
        for j in range(n):
            df_dw[j] += loss*x[i, j]
        df_db += loss
    df_dw = df_dw/m
    df_db = df_db/m

    for j in range(n):
        df_dw[j] += lambda_/m*w[j]
    return df_dw, df_db
np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
# cost_tmp = compute_cost_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
# print("Regularized cost:", cost_tmp)

# cost_tmp = compute_cost_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
# print("Regularized cost:", cost_tmp)

# dj_dw_tmp, dj_db_tmp =  compute_gradient_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

# print(f"dj_db: {dj_db_tmp}", )
# print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )

dj_dw_tmp, dj_db_tmp =  compute_gradient_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )