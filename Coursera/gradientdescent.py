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
    #number of training examples 
    m=x.shape[0] #this will gave total amount of rows that is m
    dj_dw=0
    dj_db=0 #make initial parameter equal to 0 for futute update

    for i in range(m):
        #loop 
        f_wb=w*x[i]+b #linear regresion formula 
        dj_dw_i=((f_wb-y[i])*x[i]) # gradient descent for w
        dj_db_i=f_wb-y[i] #gradient decent for b
        dj_db+=dj_db_i # sum values of b before updating 
        dj_dw+=dj_dw_i #sum values of w before updating 
    dj_dw=dj_dw/m 
    dj_db=dj_db/m

    return dj_dw, dj_db # update value 

print(compute_gradient(x_train,y_train,0,0))

# Gradient Descent computation 

def gradient_descent (x,y,w_in,b_in,alpha,num_iters,cost_function,gradient_function):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
    """
    # An array to store J and W's at each iteration primarly for graphic
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        #calculate the gradient and update the paramiteres using grident function
        dj_dw,dj_db=gradient_function(x,y,w,b)

        # update paramiters using equation for claculation minimum w and b
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        # save cost J at each interation 
        if i < 100000 :  # prevent resource exhaustion 
            J_history.append( cost_function(x,y,w,b))
            p_history.append([w,b])
        # Print cost every at intervals 10 times or as many iteration if <10
        if i% math.ceil(num_iters/10)==0:
            print(f"Iteration {i:4} : cost {J_history[-1]} ",
                  f"dj_dw: {dj_dw}  ,dj_db: {dj_db}",
                  f"w:{w},b:{b}")
    return w,b,J_history,p_history #return w and J,w  history for graphic

# initialize parameters
w_init = 0   
b_init = 0
# some gradient descent settings
iterations = 10000   # amount of iteration will be happening 
tmp_alpha = 1.0e-2   # learning rate 0,01
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})") # final result of our minimal w and b that can be used for future prediction of price 

print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")
