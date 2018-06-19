#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 15:49:52 2018

@author: ravi
"""

import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras import backend as K
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from keras import applications

###############################################################################
# image parameters
###############################################################################
img_width, img_height = 300, 300
input_shape = (img_width, img_height, 3)
input_tensor = Input(input_shape)

###############################################################################
# model 1: Using data augementation only
###############################################################################

model1 = Sequential()
model1.add(Conv2D(32, (3, 3), input_shape=input_shape))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Conv2D(32, (3, 3)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Conv2D(64, (3, 3)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Flatten())
model1.add(Dense(64))
model1.add(Activation('relu'))
model1.add(Dropout(0.5))
model1.add(Dense(1))
model1.add(Activation('sigmoid'))

model1.load_weights('final_weights/data_augmentation_weight.h5')


###############################################################################
# model 2: Using transfer learning
###############################################################################
train_data = np.load(open('training_image_features.npy'))
base_model2 = applications.VGG16(include_top=False, weights='imagenet')
model2 = Sequential()
model2.add(Flatten(input_shape=train_data.shape[1:]))
model2.add(Dense(256, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(1, activation='sigmoid'))
model2.load_weights('final_weights/transfer_learning_fc_layer_weight.h5')


###############################################################################
# model 3: with transfer learning and data augumentaion
###############################################################################
base_model3 = applications.VGG16(weights='imagenet',include_top= False, input_tensor=input_tensor)
top_model3 = Sequential()
top_model3.add(Flatten(input_shape=base_model3.output_shape[1:]))
top_model3.add(Dense(256, activation='relu'))
top_model3.add(Dropout(0.5))
top_model3.add(Dense(1, activation='sigmoid'))
model3 = Model(input= base_model3.input, output= top_model3(base_model3.output))
model3.load_weights('final_weights/transfer_learning_with_data_augmentation_weight.h5')


###############################################################################
# making prediction with trained weights
###############################################################################
def check_prediction_on_validation_data(model, base_model = None):
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
    if base_model:
        x = base_model.predict(x)
    pred = model.predict(x)[0][0]
    
    if pred > 0.5:
        print 'Predicted: ', 'no_leak', '\nActual: ', chosen_dir
    else:
        print 'Predicted: ', 'leak', '\nActual: ', chosen_dir
        
    return

check_prediction_on_validation_data(model1)
check_prediction_on_validation_data(model2, base_model2)
check_prediction_on_validation_data(model3)









