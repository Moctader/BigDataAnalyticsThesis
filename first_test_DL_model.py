# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: vscode
#     language: python
#     name: vscode
# ---

# %%
# Import Section

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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Concatenate, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, Input, Concatenate
import matplotlib.pyplot as plt
import tensorflow as tf

# %%
# Paths 
PREFIX="/Users/moctader/Thesis_code/out/pickele"

Read_data=F"{PREFIX}/samples.pkl"

# Read Data
df=gpd.GeoDataFrame(
    pd.read_pickle(Read_data),
    geometry="geometry"
)


# %%
#combine channel and label extracted

X = np.array([np.array(row['combined_channels']) for _, row in df.iterrows()])
label = np.array(df['label'])

# %%
# Find the unique channels(arrays) form the combined channels
unique_indices = [12, 10, 11, 20, 22, 23, 19,  4, 27, 24,  0,  9, 21,  3,  5, 15, 18, 6]
features = X[:, 13:37, 13:37, unique_indices]


# %%
# Assaign other features lattitude and longitude 

latitude=np.array([np.array(row['lat']) for _, row in df.iterrows()])
longitude=np.array([np.array(row['lon']) for _, row in df.iterrows()])

# %%
# Normalized the features
features = features / 255.0


# %%
# Perform train-test split with the same number of samples
X_feature_train, X_feature_test, X_scalar_train, X_scalar_test, y_train, y_test = train_test_split(
    features,
    np.column_stack((latitude, longitude)),
    label,
    test_size=0.2, random_state=42
)




# %%
# Define the CNN model for processing image features
input_feature = Input(shape=(24, 24, 18))

# Convolutional layers with increasing filters, dropout, batch normalization, and max pooling
x = Conv2D(8, (3, 3), activation='relu', dilation_rate=2)(input_feature)  # Example dilation_rate=2
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(1, 1))(x)  # Add MaxPooling
x = Dropout(0.2)(x)

x = Conv2D(16, (3, 3), activation='relu', dilation_rate=2)(x)  # Example dilation_rate=2
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(1, 1))(x)  # Add MaxPooling
x = Dropout(0.2)(x)

x = Conv2D(32, (3, 3), activation='relu', dilation_rate=2)(x)  # Example dilation_rate=2
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(1, 1))(x)  # Add MaxPooling
x = Dropout(0.2)(x)

x = Flatten()(x)

# Define the input layer for scalar values
input_scalar = Input(shape=(2,))

# Concatenate flattened features and scalar inputs
merged_input = Concatenate()([x, input_scalar])

# Additional hidden layer with fewer neurons, dropout, and batch normalization
x = Dense(64, activation='relu')(merged_input)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(32, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# Output layer
output = Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=[input_feature, input_scalar], outputs=output)

optimizers = ['sgd', 'rmsprop', 'adam', 'adagrad']

import os

# Create a directory to save results
result_directory = "results"
os.makedirs(result_directory, exist_ok=True)

# Create a file to save results
result_file_path = os.path.join(result_directory, "results.txt")
with open(result_file_path, "w") as result_file:
    for optimizer in optimizers:
        print(f"\nOptimizer: {optimizer}", file=result_file)

        # Create a new instance of the model for each optimizer to ensure the same initial weights
        model_instance = tf.keras.models.clone_model(model)

        # Compile the model with the current optimizer
        #model_instance.compile(optimizer=optimizer, loss='MSE', metrics=['accuracy'])
        model_instance.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model with early stopping
        history = model_instance.fit(
            [X_feature_train, X_scalar_train],
            y_train,
            epochs=30,
            batch_size=32,
            validation_split=0.1,
        )

        # Evaluate the model on the test set
        loss, accuracy = model_instance.evaluate([X_feature_test, X_scalar_test], y_test)
        print(f'Test Loss: {loss}', file=result_file)
        print(f'Test Accuracy: {accuracy}', file=result_file)

        # Plot learning curve
        plt.plot(history.history['accuracy'], label=f'Train Accuracy ({optimizer})')
        plt.plot(history.history['val_accuracy'], label=f'Validation Accuracy ({optimizer})')
    
    # Save the learning curves plot
    plt.title('Learning Curves for Different Optimizers')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(result_directory, 'learning_curves.png'))

# Show the learning curves for all optimizers
plt.show()



'''
epochs=30,
batch_size=32,
loss='MSE'

Optimizer: sgd
Test Loss: 0.2256123125553131
Test Accuracy: 0.6446352005004883

Optimizer: rmsprop
Test Loss: 0.2352774292230606
Test Accuracy: 0.6111587882041931

Optimizer: adam
Test Loss: 0.2707892954349518
Test Accuracy: 0.5965664982795715

Optimizer: adagrad
Test Loss: 0.23726418614387512
Test Accuracy: 0.6188841462135315

'''

'''
epochs=30,
batch_size=32,
loss='binary_crossentropy'

Optimizer: sgd
Test Loss: 0.6328206658363342
Test Accuracy: 0.6472102999687195

Optimizer: rmsprop
Test Loss: 0.668016791343689
Test Accuracy: 0.6300429105758667

Optimizer: adam
Test Loss: 0.6891878247261047
Test Accuracy: 0.6283261775970459

Optimizer: adagrad
Test Loss: 0.6503638625144958
Test Accuracy: 0.6326180100440979
'''
# %%
