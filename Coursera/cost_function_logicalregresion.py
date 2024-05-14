import numpy as np
from scipy.special import expit
import math

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

X =np.array([[0.5,1.5],[1,1],[1.5,0.5],[3,0.5],[2,2],[1,2.5]])
y = np.array([0,0,0,1,1,1])

#first variable
w_array1=np.array([1,1])
b_1= -3
# secodn variable 
w_array2=np.array([1,1])
b_2= -4

print('cost for b = -3 :',compute_cost_logistick(X,y.reshape(-1),w_array1.reshape(-1),b_1))
print('cost for b = -4 :',compute_cost_logistick(X,y.reshape(-1),w_array2.reshape(-1),b_2))
# -4 worse model for training data
