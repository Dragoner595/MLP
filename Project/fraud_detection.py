import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler 
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score 
import time 
import warnings 
warnings.filterwarnings ('ignore')

# we will use api from kagle to pull reqest for data set 

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"

# creating csv reader 

raw_data=pd.read_csv(url)
#print('There are '+ str(len(raw_data))+' observations in data set')
#print('THere are '+ str(len(raw_data.columns))+ ' varience of the columns')

# We will repeat our data set 10 times to increase amount of data for model

n_replicas = 10

# inflating original data 

big_raw_data=pd.DataFrame(np.repeat(raw_data.values,n_replicas,axis= 0),columns = raw_data.columns)

#print("There are " + str(len(big_raw_data)) + " observations in the inflated credit card fraud dataset.")
#print("There are " + str(len(big_raw_data.columns)) + " variables in the dataset.")

#print(big_raw_data.head(20))

# we create labeles class to find unique values in column what we want to predict

labels = big_raw_data.Class.unique()

# amount of unique values in the column 
sizes = big_raw_data.Class.value_counts().values

'''
This case requires special attention when training or when evaluating the quality of a model. 
One way of handing this case at train time is to bias the model to pay more attention to the samples in the minority class. 
The models under the current study will be configured to take into account the class weights of the samples at train/fit time.
'''
print("Minimal values of the data\n" , big_raw_data.min(),
      "Maximum values of the data\n" , big_raw_data.max(),
      "90 percent percentile of the data set\n",np.percentile(raw_data.Amount.values,90))


big_raw_data.iloc[:,1:30] = StandardScaler().fit_transform(big_raw_data.iloc[:,1:30])

data_matrix = big_raw_data.values

# X: feature matrix ( for this analysis, we exlude the Time variable from the dataset)

X = data_matrix[:,1:30]

# y: Labels vector 

y = data_matrix[:,30]

# data normalization

X = normalize (X , norm = 'l1')

# print the shape of the features matrix and the labels vector 

print("X.shape=", X.shape , "y.shape=" , y.shape)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=42,stratify=y)

print("X_train.shape =" , X_train.shape ,'Y_train.shape=',y_train.shape)
print("X_test.shape=", X_test.shape,"y_test.shape",y_test.shape)

