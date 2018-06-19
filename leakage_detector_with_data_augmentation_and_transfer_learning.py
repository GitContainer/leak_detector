#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 10:58:35 2018

@author: ravi
"""

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import Model
from keras.layers import Input, Dense
import os

###############################################################################
# change to the appropriate directory
###############################################################################
os.chdir('/home/ravi/Documents/leakage_detector')

###############################################################################
# specify the model parameters
###############################################################################
nb_train_samples = 128
nb_validation_samples = 32
batch_size = 16
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
img_width, img_height = 300, 300
epochs = 50
input_tensor = Input(shape=(300,300,3))

###############################################################################
# Weight initialization
###############################################################################
weights_path = '../keras/examples/vgg16_weights.h5'
top_model_weights_path = 'transfer_learning_fc_layer_weight.h5'

###############################################################################
# Model creation
###############################################################################
base_model = applications.VGG16(weights='imagenet',include_top= False,input_tensor=input_tensor)
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))
top_model.load_weights('transfer_learning_fc_layer_weight.h5')
model = Model(input= base_model.input, output= top_model(base_model.output))

###############################################################################
# Freeze the first 25 layers. For those layers weights will not be updates
###############################################################################
for layer in model.layers[:25]:
    layer.trainable = False

###############################################################################
# compile the model with a SGD/momentum optimizer and a slow learning rate
###############################################################################
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-6, momentum=0.9),
              metrics=['accuracy'])


###############################################################################
# Data augumentation configuration
###############################################################################
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

###############################################################################
# Model training to fine tune the parameters
###############################################################################
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)