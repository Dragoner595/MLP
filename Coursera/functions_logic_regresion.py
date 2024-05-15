import numpy as np
import math,copy

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

#!!!!Remember that the cost function gives you a way to measure how well a specific set of parameters fits the training data. Thereby gives you a way to try to choose better parameters.

# function of sigmoid 
# def sigmoid(x):
#  return 1 / (1 + math.exp(-x))

# Code Description 
# The algorithm for compute_cost_logistick is as follows:
# 1.Create a variable outside the loop to store the cost 
# 2. Loop over all examples in the training set.
#   2.1)Calculate z_i = w*x+b
#   2.2)Predict f_wb_i where g is the sigmoid function fw,b(x_i)=g*(z_i)
#       sigmoid is library function g(z_i)=1/1+e^-z_i
#   2.3) Calculate the loss for this example:
#       loss(fwb_(x_i),(y_i))=-y_i*log(fwb(x_i))-(1-y_i)log(1-fwb(x_i))
#   2.3.1) Add this cost to the total cost variable created outside the loop
# 3.Get the sum of cost from all iterations and return the total divided the number of examples

def compute_cost_logistick(X,y,w,b):
    """
    Compute cost

    Args:
    X (ndarray): Share (n,n) matrix of examples with n features
    y (ndarray): Share (n,) target values
    w (ndarray): Share (n) parameters for prediction 
    b (ndarray):            parameter for prediction
    Returns"
    cost (scalar): cost
    """

    m = X.shape[0] # 1
    cost = 0.0     # 1

    for i in range(m):
        z_i = np.dot(X[i],w)+b #2.1
        f_wb_i= expit(z_i)     #2.2
        cost += -y[i]*np.log(f_wb_i)-(1-y[i])*np.log(1-f_wb_i)
    
    cost = (1/m)*cost
    return cost

# Calculate the Gradient Descent 
# 1. Initialize variable to accumulate dj_dw and dj_db
# 2. For each example
#   2.1 Calculate the error for that example g(w*x_i+b)-y_i
#   2.2 for each input value Xj_i in this example,
#   2.2.1 multiply the error by the input Xj_i and add to the corresponding element of dj_dw
#   2.3 add the erro to dj_db 
# 3. Divide dj_db and dj_dw by total number of examples (m)
# 4. Note that X_i in numpy X[i,:]or X[i] and Xj_i is X[i,j]

def compute_gradient_logistic(X,y,w,b):
    """
    Compute gradient Descent

    Args:
    X : (ndarray Shape (m,n)) variable such as house size 
    y : (ndarray shape (m,))  actual values 
    w : (ndarray shape (n,))  parameters of the model
    b : (scalar)              parameter of the model 
    """

    m,n=X.shape

    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        f_wb_i= sigmoid(np.dot(X[i],w)+b)   #2.1
        err_i=f_wb_i-y[i]                   #2.1
        for j in range(n):
           dj_dw[j] = dj_dw[j] + err_i * X[i,j]    #2.2 
        dj_db = dj_db+err_i                 #2.3
    dj_dw = dj_dw / m                       #3
    dj_db = dj_db / m                       #3

    return dj_db,dj_dw #index dj_db to return scalat value 

def gradient_descent(X,y,w_in,b_in,alpha,num_iters):
   """
   Perform batch gradient descent

   Args:
   X (ndarray): Shape (m,n)         matrix of examples
   y (ndarray): Shape (n,)          target value of each example
   w_in (ndarray): Shape (n,)       Initial values of parameters of the model
   b_in (scalar)                    Initoal value of parameters of the mdoel
   alpha(float)                     Learning rate
   num_iters(int)                   number of iteration to run gradient descent

   Retirns:
   w (ndarray): Shape(n,)           Updated values of parameters
   b (scalar)                       Updated values of parameters
   """
   # number of training examples 
   m = len(X)

   # An array to store cost J and w's at each iteration primarly for graphing later
   J_history = []
   w = copy.deepcopy(w_in) #avoid modifying global w with in function
   b = b_in

   for i in range(num_iters):
      # Calculate the gradient and update the parameter
      dj_db,dj_dw = compute_gradient_logistic(X,y,w,b)

      # Update Parameters using w,b, alpha and gradient

      w = w - alpha*dj_dw
      b = b - alpha*dj_db

      # Save cost J at each iterations 
      if i<100000:          #prevent resource exhausting
        J_history.append(compute_cost_logistick(X,y,w,b))
      # Print cost every at intervals q0 times or as many iterations if <10
      if i% math.ceil(num_iters/10)==0:
         print(f'Iteration {i:4d}; Cost {J_history[-1]} ')
   return w,b,J_history #return final w,b and J_jistory for graphic 