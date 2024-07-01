import os
import zipfile
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(3,activation = 'sigmoid', input_dim= 2))
model.add(Dense(1))
model.summary()

base_dir= '/workspaces/MLP/Project/cats_and_dogs_filtered'

train_dir =os.path.join(base_dir,'train')

test_dir = os.path.join(base_dir,'validation')
#directory with our training cat picture 
train_cats_dir = os.path.join(train_dir,'cats')

#directory with our training dog picture 
training_dogs_dir = os.path.join(train_dir,'dogs')

#directory wit our test cat pictures
test_cats_dir= os.path.join(test_dir,'cats')

#Directory with our validation dog pictures

test_dogs_dir= os.path.join(test_dir,'dogs')

print("Ammount of pictires in the training foulder cats",len(os.listdir(train_cats_dir)))
# we used Len function to count amount of the files in the foulder
print("Ammount of pictires in the training foulder dogs",len(os.listdir(training_dogs_dir)))
print("Ammount of pictires in the test foulder cats",len(os.listdir(test_cats_dir)))
print("Ammount of pictires in the test foulder dogs",len(os.listdir(test_dogs_dir)))

from tensorflow.keras.preprocessing.image import ImageDataGenerator 

#All image will be rescaled by image generator 
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator ( rescale = 1./ 255)

# Flow training images in batches of using train_datagen generator 
train_generator = train_datagen.flow_from_directory(train_dir,traget_size = (150,150)
                                                    target_size = (150,150)
                                                    batch_size = 20 ,
                                                    class_mode = 'binary'
                                                    )

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = train_datagen.flow_from_directory(train_dir,target_size = (150,150),
                                                         batch_size=20,
                                                         calss_mode = 'binary')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = train_datagen.flow_from_directory(test_dir,target_size = (150,150),
                                                         batch_size=20,
                                                         calss_mode = 'binary')