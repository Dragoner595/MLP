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

"""
The image pixel values range between 0 to 255, thus to normalize the images we have to use ImageDataGenerator class which will genrate batches of normalized input images based out of the train and validation directory. 
Target size of the image selected as (150,150) but it can be rescaled later. 
Since this project only deals with only two class (Cat and Dog) of images so the class_mode selected as 'binary' for generating the flow of training and validation images.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by ImageGenerator 

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen =  ImageDataGenerator(rescale = 1. /255)

# Generating training image with batches of 20 
# image clasification deals only with cat and dogs so we select class mode binary 
train_generator = train_datagen.flow_from_directory(
    train_dir,target_size = (150,150),
    batch_size = 16, 
    class_mode = 'binary'
    )

# Copy same code jsut change to test sample 
validation_generator = test_datagen.flow_from_directory(
    test_dir,target_size = (150,150),
    batch_size = 16, 
    class_mode = 'binary')

from tensorflow.keras.optimizers import RMSprop 

model_1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,input_shape = [150,150,3],kernel_size = (3,3),activation = 'relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
    # not adding imput shape becauze provide shape in original layer of model 
    tf.keras.layers.Conv2D(128,kernel_size = (3,3),activation = 'relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
    # creating flatten layer to transfer from 2d feature extractopn map onto one dimensional 
    tf.keras.layers.Flatten(),
    #creating Dense layer it will have 256 neirons 
    tf.keras.layers.Dense(256,activation = 'relu'),
    # last layer will ahve only one neuron because of binary clasification , it will have only one vector of probability to what class it belongs too 
    tf.keras.layers.Dense(1,activation = 'sigmoid'),
                                     ])

model_1.summary()
model_1.compile(loss='binary_crossentropy',optimizer=RMSprop(learning_rate=1e-4),metrics = ['acc'])

#now we will train the model 
train_model = model_1.fit(
    train_generator,steps_per_epoch = 100, # 2000 images = batch_size*steps
    epochs = 25,
    validation_data = validation_generator,
    validation_steps = 50, # 1000 images = batch_size*steps
    verbose = 2)