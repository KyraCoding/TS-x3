import os
import tensorflow as tf
import matplotlib.pyplot as plt


print("TensorFlow version:", tf.__version__)

tf.keras.backend.set_image_data_format('channels_last')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


train_dir = 'training'
test_dir = 'testing'

def divvy(training, testing):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    for name in os.listdir(train_dir):
        for file in os.listdir(os.path.join(train_dir, name)):
            os.remove(os.path.join(train_dir, name, file))
    for name in os.listdir(test_dir):
        for file in os.listdir(os.path.join(test_dir, name)):
            os.remove(os.path.join(test_dir, name, file))
    for name in os.listdir("data"):
        os.makedirs(os.path.join(train_dir, name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, name), exist_ok=True)
        counter = 0
        for file in os.listdir(os.path.join("data/", name)):
            if (counter%(training+testing) < training):
                shutil.copy(os.path.join("data/", name, file), os.path.join(train_dir, name, file))
            else:
                shutil.copy(os.path.join("data/", name, file), os.path.join(test_dir, name, file))
            counter+=1
# Divide data into testing and training
divvy(7,3)


train_datagen = train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(450, 500), batch_size=16,class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir,target_size=(450, 500),batch_size=16,class_mode='categorical')

model = Sequential()
model.add(Input(shape=(450, 500, 3)))

model.add(Conv2D(32, (3, 3), activation='relu',padding='same',kernel_regularizer=l2(0.001)))
BatchNormalization()
model.add(MaxPooling2D((2, 2)))
Dropout(0.15)

model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_regularizer=l2(0.001)))
BatchNormalization()
model.add(MaxPooling2D((2, 2)))
Dropout(0.15)

model.add(Conv2D(128, (3, 3), activation='relu',padding='same',kernel_regularizer=l2(0.001)))
BatchNormalization()
model.add(MaxPooling2D((2, 2)))
Dropout(0.15)

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(train_generator,epochs=200,validation_data=test_generator, callbacks=[early_stopping])

#model.save('models/model.keras')

def smooth_curve(points, factor=0.8):
    smoothed = []
    for point in points:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return smoothed

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()