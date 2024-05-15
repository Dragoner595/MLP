import numpy as np
import math, copy
from cost_function_logicalregresion import compute_cost_logistick
# Calculate the Gradient Descent 
# 1. Initialize variable to accumulate dj_dw and dj_db
# 2. For each example
#   2.1 Calculate the error for that example g(w*x_i+b)-y_i
#   2.2 for each input value Xj_i in this example,
#   2.2.1 multiply the error by the input Xj_i and add to the corresponding element of dj_dw
#   2.3 add the erro to dj_db 
# 3. Divide dj_db and dj_dw by total number of examples (m)
# 4. Note that X_i in numpy X[i,:]or X[i] and Xj_i is X[i,j]

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

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

X_tmp=np.array([[0.5,1.5],[1,1],[1.5,0.5],[3,0.5],[2,2],[1,2.5]])
y_tmp= np.array([0,0,0,1,1,1])
w = np.array([2.,3.])
b = 1.
#dj_db,dj_dw = compute_gradient_logistic(X_tmp,y_tmp,w,b)
#print(f'dj_dw,non_vectorized version:{dj_dw}')
#print(f'dj_db,non_vectorized version:{dj_db}')

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
    
w_in = np.zeros_like(X_tmp[0])
b_in = 0.
alpha = 0.1
num_iters = 10000

w_out,b_out,_=gradient_descent(X_tmp,y_tmp,w_in,b_in,alpha,num_iters)
print(f'\nupdated parameters: w:{w_out} , b:{b_out}')

m,_=X_tmp.shape
for i in range(m):
    print(f'prediction: {np.dot(X_tmp[i],w_out)+b_out:0.2f},target value: {y_tmp[i]}')