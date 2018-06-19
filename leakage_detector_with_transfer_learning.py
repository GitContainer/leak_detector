#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 09:45:37 2018

@author: ravi
"""

import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.optimizers import Adam
from keras import applications
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


###############################################################################
# change to the appropriate directory
###############################################################################
os.chdir('/home/ravi/Documents/leakage_detector')


###############################################################################
# Specify the parameters
###############################################################################
nb_train_samples = 128
nb_validation_samples = 32
batch_size = 16
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
img_width, img_height = 300, 300

###############################################################################
# Import the pretrained VGG16 model without the top layer
###############################################################################
model = applications.VGG16(include_top=False, weights='imagenet')

###############################################################################
# Create the generator to read the image files
###############################################################################
datagen = ImageDataGenerator(rescale=1. / 255)
generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

###############################################################################
# Get and save the training and the validation data features. Their features 
# are the VGG16 model output
###############################################################################

training_image_features = model.predict_generator(generator, nb_train_samples // batch_size)

np.save(open('training_image_features.npy', 'w'), training_image_features)

generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

validation_image_features = model.predict_generator(generator, nb_validation_samples // batch_size)

np.save(open('validation_image_features.npy', 'w'), validation_image_features)

###############################################################################
# Load the training and validation image features which will act as an input to
# our model. Create their corresponding labels.
###############################################################################

train_data = np.load(open('training_image_features.npy'))
train_labels = np.array([0] * 64 + [1] * 64)

validation_data = np.load(open('validation_image_features.npy'))
validation_labels = np.array([0] * 16 + [1] * 16)

###############################################################################
# Create our own model that will take as input the image features using the 
# VGG16 model
###############################################################################

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(Adam(lr=1e-6), loss='binary_crossentropy', metrics=['accuracy'])

###############################################################################
# Train the model
###############################################################################
model.fit(train_data, train_labels,
          epochs=100,
          batch_size=batch_size,
          validation_data=(validation_data, validation_labels))

model.save_weights('transfer_learning_fc_layer_weight.h5')

###############################################################################
# Reading model parameters from a file
###############################################################################
model.load_weights('transfer_learning_fc_layer_weight.h5')

###############################################################################
# making prediction by choosing a random image from the validation file
###############################################################################
def check_prediction_on_validation_data():
    dir_name = '/home/ravi/Documents/leakage_detector/data/validation/'
    class_names = os.listdir(dir_name)
    chosen_dir = np.random.choice(class_names)
    chosen_i = np.random.choice(range(1,17))
    image_name = chosen_dir + '%03d.jpg' %(chosen_i)
    full_name = dir_name + chosen_dir + '/' + image_name
    img = load_img(full_name)
    plt.imshow(img)
    x = img_to_array(img)
    x = x/255
    x = x.reshape((1,) + x.shape)
    int_model = applications.VGG16(include_top=False, weights='imagenet')
    model_input = int_model.predict(x)
    pred = model.predict_classes(model_input)[0][0]
    
    for key, value in validation_generator.class_indices.items():
        if pred == value:
            print 'Predicted: ', key, '\nActual: ', chosen_dir
            break
        if pred == value:
            print 'Predicted: ', key, '\nActual: ', chosen_dir
            break
    

check_prediction_on_validation_data()
