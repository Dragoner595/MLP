import numpy as np
import pandas as pd 
import time

# create a random data set with data 
rows = 10000000
cols = 4

housing_data=np.random.randint(500,2500,size=(rows,cols-3))
housing_data = np.column_stack((housing_data, np.random.randint(1, 10, size=(rows,cols-2))))
housing_data = np.column_stack((housing_data, np.random.randint(10, 50, size=rows)))

# make dot product multiplication 
def dot_prod(a,b):

    x=0

    for i in range (a.shape[0]):
        x=x+a[i]*b[i]

    return x

a=housing_data[:,1]
b=housing_data[:,2]
tic=time.time()
c=dot_prod(a,b)
toc=time.time()
print(f"dot_prod(a,b)={c:.4f}")
print(f"time for function dot= {1000*(toc-tic):.4f}ms")
tic=time.time()
c=np.dot(a,b)
toc=time.time()
print(f"Vectorized version={c}")
print(f"Vectorized time= {1000*(toc-tic):.4f}ms")
