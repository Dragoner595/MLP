import numpy as np 
import copy
import math
import pandas as pd 


X_train=pd.read_csv('/workspaces/MLP/X_train.csv',usecols=[0])
y_train=pd.read_csv('/workspaces/MLP/y_train.csv')
X_train=X_train.drop[0]
print(X_train)
