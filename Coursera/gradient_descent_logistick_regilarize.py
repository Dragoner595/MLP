import numpy as np 
import math 
from functions_logic_regresion import compute_gradient_logistic

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def compute_gradient_logistic_reg(X,y,w,b,lambda_):
    """
    Compute gradient Descent

    Args:
    X : (ndarray Shape (m,n)) variable such as house size 
    y : (ndarray shape (m,))  actual values 
    w : (ndarray shape (n,))  parameters of the model
    b : (scalar)              parameter of the model 
    lamda_(scalar)            Controls amount of regularization 

    Return:
    dj_dw (ndarray Shape (n,)): The gradient of the cost w.r.t the parameters w
    dj_db(scalar)             : The gradient of the cost w.r.t the parameter b
    """
    m,n=X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w)+b)
        err_i = f_wb_i-y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i*X[i,j]
        dj_db = dj_db + err_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m 

    for j in range(n):
        dj_dw[j] = dj_dw[j]+(lambda_/m)*w[j] # if forgot check your notebook to see how it decreasing w value 

    return dj_db,dj_dw

X_tmp=np.random.rand(5,3)
y_tmp= np.array([0,1,0,1,0])
w = np.random.rand(X_tmp.shape[1])
b = 0.5
lambda_tmp=0.7

dj_db,dj_dw = compute_gradient_logistic_reg(X_tmp,y_tmp,w,b,lambda_tmp)
print(f'dj_dw,Regularized:{dj_dw}')
print(f'dj_db:{dj_db}')

dj_db,dj_dw = compute_gradient_logistic(X_tmp,y_tmp,w,b)
print(f'dj_dw,Not-Regularized:{dj_dw}')
print(f'dj_db:{dj_db}')