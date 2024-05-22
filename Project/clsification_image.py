import os
import zipfile
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
# create model with 3 neurons , and 2 output dimensions )
model.add(Dense(3,activation = 'sigmoid',input_dim=2))
# create an output layerr 
model.add(Dense(1))

# Printing our summery of our model 
model.summary()

local_zip ='/workspaces/MLP/Project/cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(local_zip,'r')
zip_ref.extractall('/workspaces/MLP/Project/')
zip_ref.close()

base_dir = '/workspaces/MLP/Project/cats_and_dogs_filtered'

train_dir = os.path.join(base_dir,'train')
test_dir = os.path.join(base_dir,'validation')

# Directory with our training cat pictures 

train_cats_dir = os.path.join(train_dir,'cats')

# Directory with our training dog pictures 

train_dogs_dir = os.path.join(train_dir,'dogs')

# Directory with our test cat pictures

test_cats_dir = os.path.join(test_dir,'cats')

# Directory with our validation dog pictures 

test_dogs_dir = os.path.join(test_dir,'dogs')

# cheking ammount of the cat and dog images in train and validat folders 

print("Number of cat images in train folder:",len(os.listdir(train_cats_dir)))
print("Number of dog images in train folder:",len(os.listdir(train_dogs_dir)))
print("Number of cat images in validation folder:", len(os.listdir(test_cats_dir)))
print("Number of dog images in validation folder:", len(os.listdir(test_dogs_dir)))

