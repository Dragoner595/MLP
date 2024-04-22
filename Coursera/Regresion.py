import numpy as np
import pandas as pd 

# create a random data set with data 
rows = 100
cols = 4

housing_data=np.random.randint(500,2500,size=(rows,cols-3))
housing_data = np.column_stack((housing_data, np.random.randint(1, 10, size=(rows,cols-2))))
housing_data = np.column_stack((housing_data, np.random.randint(10, 50, size=rows)))

