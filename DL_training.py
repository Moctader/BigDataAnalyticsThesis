# %%
import geopandas as gpd
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense

# %%
SAMPLES_PICKLE = "/Users/akusok/wrkdir/Md-dataset/samples.pkl"

# %%
# Read Data
df=gpd.GeoDataFrame(pd.read_pickle(SAMPLES_PICKLE), geometry="geometry")

# %%
# select Feature and label
X = np.array([np.array(row['combined_channels']) for _, row in df.iterrows()])
label = np.array(df['label'])

# %%
#select only the unique features
# unique_arrays, unique_indices = np.unique(X, axis=-1, return_index=True)

# assign into variable
# feature=unique_arrays

unique_indices = [12, 10, 11, 20, 22, 23, 19,  4, 27, 24,  0,  9, 21,  3,  5, 15, 18, 6]
feature = X[:, 21:30, 21:30, unique_indices]

# %%
# splitting
X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=42)

# %%

# Define the input shape
input_shape = (9, 9, 18)

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


# %%

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# %%



