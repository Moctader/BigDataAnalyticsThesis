# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import glob
import os
import math
import csv
import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Input, Flatten, Dense
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense


# %%


# %%
df=gpd.GeoDataFrame(
    pd.read_pickle("/Users/moctader/Thesis_code/pickle/samples.pkl"),
    geometry="geometry"
)

# %%
X = np.array([np.array(row['combined_channels']) for _, row in df.iterrows()])
label = np.array(df['label'])


# %%
unique_arrays, unique_indices = np.unique(X, axis=-1, return_index=True)


# %%
feature=unique_arrays

# %%
X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=42)


# %%

# Define the input shape
input_shape = (50, 50, 18)

# Define the input layer
input_layer = Input(shape=input_shape)

# Convolutional layers
conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(conv1)

# Flatten the output from convolutional layers
flattened_input = Flatten()(conv2)

# Dense layers
dense1 = Dense(512, activation='relu')(flattened_input)

# Output layer for binary classification with sigmoid activation
output_layer = Dense(units=1, activation='sigmoid')(dense1)

# Create the model
model = keras.Model(inputs=input_layer, outputs=output_layer)

# Compile the model for binary classification
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))



# %%



