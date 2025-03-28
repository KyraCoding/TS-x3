import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf


print("TensorFlow version:", tf.__version__)

tf.keras.backend.set_image_data_format('channels_last')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from livelossplot import PlotLossesKeras

train_dir = 'training'
test_dir = 'testing'

def divvy(training=8, testing=2):
    for name in os.listdir(train_dir):
        for file in os.listdir(os.path.join(train_dir, name)):
            os.remove(os.path.join(train_dir, name, file))
    for name in os.listdir(test_dir):
        for file in os.listdir(os.path.join(test_dir, name)):
            os.remove(os.path.join(test_dir, name, file))
    for name in os.listdir("data"):
        counter = 0
        for file in os.listdir(os.path.join("data/", name)):
            if (counter%(training+testing) < training):
                shutil.copy(os.path.join("data/", name, file), os.path.join(train_dir, name, file))
            else:
                shutil.copy(os.path.join("data/", name, file), os.path.join(test_dir, name, file))
            counter+=1
# Divide data into testing and training
divvy()


train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(450, 500), batch_size=32,class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir,target_size=(450, 500),batch_size=32,class_mode='categorical')
                                                                                               
model = Sequential()
model.add(Input(shape=(450, 500, 3))) 
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(train_generator,epochs=50,validation_data=test_generator)      

model.save('models/model.keras')              