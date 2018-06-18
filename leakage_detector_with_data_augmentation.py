#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 13:11:57 2018

@author: ravi
"""

import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.optimizers import Adam
import matplotlib.pyplot as plt



###############################################################################
# change to the appropriate directory
###############################################################################
os.chdir('/home/ravi/Documents/leakage_detector')


###############################################################################
# Specify the parameters
###############################################################################
img_width, img_height = 300, 300
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 128
nb_validation_samples = 32
epochs = 5
batch_size = 16


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
###############################################################################
# Create the deep covnet model
###############################################################################
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(Adam(lr=1e-6), 
              loss='binary_crossentropy',
              metrics=['accuracy'])

###############################################################################
# Specifying data augumentation parameters for training and testing data.
# The test data is only scaled
###############################################################################
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

###############################################################################
# Module to generate new data
###############################################################################

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

###############################################################################
# Model training using augmented dataset
###############################################################################
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('data_augmentation_weight.h5')

###############################################################################
# making prediction
###############################################################################
dir_name = '/home/ravi/Documents/leakage_detector/data/train/leak/'
for i in range(1,64):
    image_name = 'leak%03d.jpg' %(i)
    full_name = dir_name + image_name
    print full_name
    img = load_img(full_name)
    x = img_to_array(img)
    x = x/255
    x = x.reshape((1,) + x.shape)
    print model.predict_classes(x)
    
dir_name = '/home/ravi/Documents/leakage_detector/data/train/no_leak/'
for i in range(1,64):
    image_name = 'no_leak%03d.jpg' %(i)
    full_name = dir_name + image_name
    print full_name
    img = load_img(full_name)
    x = img_to_array(img)
    x = x/255
    plt.figure()
    plt.imshow(x)
    x = x.reshape((1,) + x.shape)
    print model.predict(x)



###############################################################################
# Model parameters
###############################################################################
model.get_weights()








