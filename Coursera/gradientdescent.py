import math, copy
import numpy as np
import matplotlib.pyplot as plt

x_train= np.array([1.0,2.0])
y_train=np.array([300.0,500.0])

def compute_cost(x,y,w,b):
    m=x.shape[0]
    cost=0

    for i in range(m):
        f_wb=w*x[i]+b
        cost=cost+(f_wb-y[i])**2
        total_cost=1/(2*m)*cost

        return total_cost
    
def compute_gradient(x,y,w,b):
    """
    Compute the gradient for linear regression 
    Args:
        x(ndarray(m,)):Data, m examples 
        y(ndarray (m,)):target values 
        w,b (scalar) : model parameters
    Returns :
        dj_dw (scalar): The gradient of the cost w.r.t the parametes w
        dj_db (scalar): The gradient of the cost w.r.t. the parameter b
    """
    #number of training wxamples 
    m=x.shape[0] #this will gave total amount of rows that is m
    dj_dw=0
    dj_db=0 #make initial parameter equal to 0 for futute update

    for i in range(m):
        #loop 
        f_wb=w*x[i]+b #linear regresion formula 
        dj_dw_i=(f_wb-y[i]*x[i]) # gradient descent for w
        dj_db_i=f_wb-y[i] #gradient decent for b
        dj_db+=dj_db_i # sum values of b before updating 
        dj_dw+=dj_dw_i #sum values of w before updating 
    dj_dw=dj_dw/m 
    dj_db=dj_db/m

    return dj_dw, dj_db # update value 

print(compute_gradient(x_train,y_train,-549.4,-299.4))
