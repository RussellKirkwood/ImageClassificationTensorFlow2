from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

import os
import numpy as np

batch_size = 128
epochs = 1
IMG_HEIGHT = 150
IMG_WIDTH = 150

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
train_dir = os.path.join(PATH, 'train')
train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('my_model.h5')

# Show the model architecture
new_model.summary()

#this assumes image is in directory of script. you may get image form anywhere
imagename = 'dog.45.jpg' 
# Get test image ready
test_image = image.load_img(imagename, target_size=(IMG_WIDTH, IMG_HEIGHT, 3))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

predictionFromModel = new_model.predict(test_image)
predictionFromModelClass = new_model.predict_classes(test_image)

labels = (train_data_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())

print(predictionFromModel)
print(predictionFromModelClass)

print (labels)
