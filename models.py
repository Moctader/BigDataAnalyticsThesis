import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Set your data directories
ass_dir = '/Users/moctader/Thesis_code/output20/landsat/ASS/'
non_ass_dir = '/Users/moctader/Thesis_code/output20/landsat/non-ASS'

# Image dimensions and other parameters
img_size = (50, 50)
batch_size = 32

# Data augmentation and normalization for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Load and augment the training data
train_generator = train_datagen.flow_from_directory(
    '/Users/moctader/Thesis_code/output20/landsat/',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Create the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Split the data into training and testing sets
train_set, test_set = train_test_split(train_generator, test_size=0.2, random_state=42)

# Train the model
model.fit(train_set, epochs=10, validation_data=test_set)
