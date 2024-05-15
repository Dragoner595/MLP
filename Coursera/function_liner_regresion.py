import copy,math
import numpy as np
import matplotlib.pyplot as plt 


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
        err=(np.dot(X[i],w)+b)-y[i] # we find X[0] == [ 2104,5,1,45] with calculation of w==[0.39133535,18.75376741,-53.36032453,-26.42131618] substract y[0]==[460]
        #print(f"update of err before loop {err}")
        for j in range(n):
            #print(f"update of w before loop {dj_dw}")
            dj_dw[j]=dj_dw[j]+err*X[i,j] # after we find for X[0] we move to loop for each of the column calculation in a row X[0]
            #it will be at first 0[0] = 0[0] + our calcualtion for X[0] multiply by X[0,0] and record it and do same calculation but for X[0,1] and till X[3,4]
        dj_db=dj_db+err
        #print(f"update of err {err}")
        #print(f"update of w {dj_dw}")
    #print(f"update of w before substraction of m: {dj_dw}")
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    

    return dj_db,dj_dw

def gradient_descent (X,y,w_in,b_in,cost_function,gradient_function,alpha,num_iters):
  """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
    """
  
  # An array to store cost J nad w's at each iteration primarly for graphic representation 
  J_hist = []
  w = copy.deepcopy(w_in) #avoid modifying global w within function 
  b = b_in


  for i in range (num_iters):
      #calculate the gradient and update the parameter 
      dj_db,dj_dw = gradient_function(X,y,w,b)  #None

      #Update parameters using w,b, alpha and gradient 
      w = w - alpha*dj_dw       #None
      b = b - alpha*dj_db       #None

      #Save cost j at ech iteration 
      if i <100000:         #prevent resource exhausting
          J_hist.append (cost_function(X,y,w,b))
      
      if i % math.ceil(num_iters/10)==0:
          print(f'Iteration {i:4d}: Cost {J_hist[-1]:8.2f}')
  
  return w,b,J_hist #return final w,b and J_history for graphic

