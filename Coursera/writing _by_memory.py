import numpy as np 
import pandas as pd 

def computing_cost(X,y,w,b)
    
    cost=0
    m = x.shape[0]

    if i in range(m):

        f_xb = np.dot(X[i],y)+b

        cost=cost + (f_xb -y[i])**2
    
    total_cost= 1 / (2*m)*cost
    return

def compute_gradient(X,y,w,b):
    dx_dw=[0,0,0,0]
    dy_db=0
    m,n=x.shape

    for i in range(m):
        err=(np.dot(X[i],w)+b)-y[i]

        for j in range(n):
            dx_dw[j]= dx_dw[j] + err*X[i,j]
        
        dj_db = dj_db+err

    dj_db = dj_db/m
    dj_dw = dj_dw/m
    return dj_db,dj_db


def compute_gradient_descent(X_train,y_train,w,b,alpha,compute_cost,compute_gradient):

    J_hist=[]
    



