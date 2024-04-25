import copy,math
import numpy as np
import matplotlib.pyplot as plt 

X_train=np.array([[2104,5,1,45],[1416,3,2,40],[852,2,1,35]])
y_train=np.array([460,232,178])


print(f"X_shape: {X_train.shape}, X type:{type(X_train)}")
print(X_train)
print(f'y_train: {y_train.shape},y type:{type(y_train)}')
print(y_train)

b_init= 785.1811367994083
w_init=np.array([0.39133535,18.75376741,-53.36032453,-26.42131618])

def predict_single_loop(x,w,b):
    """
    single predict using linear regression
    
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters    
      b (scalar):  model parameter     
      
    Returns:
      p (scalar):  prediction
    """
    n = x.shape[0]
    p = 0
    for i in range(n):
        p_i=x[i]*w[i]
        p=p+p_i
    p=p+b
    return p

# get a row from our training data 
x_vec=X_train[0,:]
print(f"x_vec shape {x_vec.shape} , x_vec value: {x_vec}")

# make a prediction 
f_wb = predict_single_loop(x_vec, w_init,b_init)
print(f"f_wb_shape {f_wb.shape}, prediction loop: {f_wb}")

# will performe same calculation but with dot product function 

def prediction(x,w,b):
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters   
      b (scalar):             model parameter 
      
    Returns:
      p (scalar):  prediction
    """
    p=np.dot(x,w)+b
    return p 

# will use same x_vec from before and create new vector f_wb_dot
f_wb_dot=prediction(x_vec,w_init,b_init)

print(f"F_wb_vector: {f_wb_dot.shape}, prediction dor porduct : {f_wb_dot}")

# compute cost with multiple variables 

def compute_cost (X,y,w,b):
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost=0

    for i in range(m):
        f_wb_i = np.dot(X[i],w)+b
        cost = cost+(f_wb_i-y[i])**2
    cost=cost/(2*m)
    return cost 

# Compute and display cost using our pre-chosen optimal parameters. 
cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')


# compute gradient with multiple values 

def compute_gradient(X,y,w,b):
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n=X.shape
    dj_dw=np.zeros((n,))
    dj_db=0.

    for i in range(m):
        err=(np.dot(X[i],w)+b)-y[i] # we find X[0] == [ 2104,5,1,45] with calculation of w==[0.39133535,18.75376741,-53.36032453,-26.42131618] substract y[460]
        for j in range(n):
            print(f"update of w before loop {dj_dw}")
            dj_dw[j]=dj_dw[j]+err*X[i,j] # after we find for X[0] we move to loop for each of the column calculation in a row X[0]
            #it will be at first 0[0] = 0[0] + our calcualtion for X[0] multiply by X[0,0] and record it and do same calculation but for X[0,1] and till X[3,4]
        dj_db=dj_db+err
        print(f"update of w {dj_dw}")
    print(f"update of w before substraction of m: {dj_dw}")
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    

    return dj_db,dj_dw

#Compute and display gradient 
tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')