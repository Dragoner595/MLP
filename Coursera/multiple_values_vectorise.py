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
    