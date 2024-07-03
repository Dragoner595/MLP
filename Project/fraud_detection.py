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
#print("Minimal values of the data\n" , big_raw_data.min(),"Maximum values of the data\n" , big_raw_data.max(),"90 percent percentile of the data set\n",np.percentile(raw_data.Amount.values,90))


big_raw_data.iloc[:,1:30] = StandardScaler().fit_transform(big_raw_data.iloc[:,1:30])

data_matrix = big_raw_data.values

# X: feature matrix ( for this analysis, we exlude the Time variable from the dataset)

X = data_matrix[:,1:30]

# y: Labels vector 

y = data_matrix[:,30]

# data normalization

X = normalize (X , norm = 'l1')

# print the shape of the features matrix and the labels vector 

#print("X.shape=", X.shape , "y.shape=" , y.shape)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=42,stratify=y)

#print("X_train.shape =" , X_train.shape ,'Y_train.shape=',y_train.shape)
#print("X_test.shape=", X_test.shape,"y_test.shape",y_test.shape)


# It takes into account the class imbalance present in this dataset 
# we finding needed weight for our prediction model 
w_train = compute_sample_weight('balanced',y_train) 

#import the dicision tree Classifier model from scikit learn 

from sklearn.tree import DecisionTreeClassifier 

# for reproducible output across multiple function calls ,set random_state to a given integer value 

sklearn_dt = DecisionTreeClassifier(max_depth = 4,random_state= 35 )

# train a dicision tree Clasifier using scihit-learn 
t0 = time.time() # print ting us training time spend for training 

sklearn_dt.fit(X_train,y_train,sample_weight = w_train)

sklearn_time = time.time()-t0
#print("[Scikit-Learn] Training time (s):  {0:.5f}".format(sklearn_time))

"""
if not already computed, 
compute the sample weights to be used as input to the train routine so that 
it takes into account the class imbalance present in this dataset
w_train = compute_sample_weight('balanced', y_train)
"""

# importing the Decision Tree Classifier Model from Snap ML
from snapml import DecisionTreeClassifier 

"""
Snap ML offers multi-threaded CPU/GPU training of decision trees, unlike scikit-learn
to use the GPU, set the use_gpu parameter to True 
snapml_dt = DecisionTreeClassifier(max_depth=4, random_state=45, use_gpu=True)
"""


# to set the number of CPU threads used at training time, set the n_jobs parameter
# for reproducible output across multiple function calls, set random_state to a given integer value

snapml_dt = DecisionTreeClassifier(max_depth=4, random_state=45, n_jobs=4)

#train dicision tree clasifier model using SNap ml
t0 = time.time()

snapml_dt.fit(X_train,y_train,sample_weight = w_train)
snapml_time = time.time()-t0
#print("[Snap ML] Training time (s):  {0:.5f}".format(snapml_time))

training_speedup = sklearn_time/snapml_time

#print('[Decision Tree Classifier] Snap ML vs Scikit-Learn speedup: {0:.2f}x'.format(training_speedup))

# run inference and compute the probabilities of the test samples 
# to belong to the class of fraudulent transactions

sklearn_pred = sklearn_dt.predict_proba(X_test)[:,1]

# evaluate the Compute Area Under the Receiver Operating Characteristic 
# Curve (ROC-AUC) score from the predictions

sklearn_roc_auc = roc_auc_score(y_test, sklearn_pred)

print('[Scikit-Learn] ROC-AUC score : {0:.3f}'.format(sklearn_roc_auc))

# run inference and compute the probabilities of the test samples
# to belong to the class of fraudulent transactions

snapml_pred = snapml_dt.predict_proba(X_test)[:,1]

# evaluate the Compute Area Under the Receiver Operating Characteristic
# Curve (ROC-AUC) score from the prediction scores

snapml_roc_auc = roc_auc_score(y_test, snapml_pred)   
print('[Snap ML] ROC-AUC score : {0:.3f}'.format(snapml_roc_auc))

# import the linear Support Vector Machine (SVM) model from Scikit-Learn

from sklearn.svm import LinearSVC

# instantive a scikit-learn  SVM Model 
# to indicate the class imbalance at fit time , set class_weight = 'balance'
# for reproducible output across multiple function calls, set random_state to a given integer values 
sklearn_svm = LinearSVC(class_weight = 'balanced', random_state = 31 , loss = 'highe', fit_intercept = False )

# train a linear Support Vector Machine model using Scikit-Learn 

t0 = time.time()
sklearn_svm.fit(X_train,y_train)
sklearn_time = time.time()- t0

print("[Scikit-Learn] Training time (s):  {0:.2f}".format(sklearn_time))