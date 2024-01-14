# %%
import geopandas as gpd
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense


# %%
PREFIX = "/Users/moctader/Thesis_code"  # folder with files
PREFIX = "/Users/akusok/wrkdir/Golam"  # folder with files

# %%
data_path = f"{PREFIX}/samples.pkl"

# %%
# Read Data
df=gpd.GeoDataFrame(
    pd.read_pickle(data_path),
    geometry="geometry"
)

# %%
# select Feature and label
X = np.array([np.array(row['combined_channels']) for _, row in df.iterrows()])
label = np.array(df['label'])

# %%
# select only the unique features
# unique_arrays, unique_indices = np.unique(X, axis=-1, return_index=True)
# feature=unique_arrays

# same thing but without waiting
unique_indices = [12, 10, 11, 20, 22, 23, 19,  4, 27, 24,  0,  9, 21,  3,  5, 15, 18, 6]
feature = X[:, :, :, unique_indices]

# %%
# splitting
X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=42)

# %% [markdown]
# ## Dynamic learning rate - decreases over time

# %%
def make_model_dynamic_lr(scale=1.0):
    # Define the input shape
    input_shape = (50, 50, 18)

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Convolutional layers
    # add 1 neuron to have at least 1 neuron with small scale
    conv1 = Conv2D(int(32 * scale) + 1, kernel_size=(3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(int(64 * scale) + 1, kernel_size=(3, 3), activation='relu')(conv1)

    # Flatten the output from convolutional layers
    flattened_input = Flatten()(conv2)

    # Dense layers
    dense1 = Dense(int(512 * scale), activation='relu')(flattened_input)

    # Output layer for binary classification with sigmoid activation
    output_layer = Dense(units=1, activation='sigmoid')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    learning_rate_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=100,
        decay_rate=0.94,
    )

    # Compile the model for binary classification
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate_schedule), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# %%
model_dynamic = make_model_dynamic_lr(scale=0.3)
model_dynamic.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 146/146 [==============================] - ETA: 0s - loss: 183.0325 - accuracy: 0.5559
# 2024-01-11 16:24:47.328704: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
# 146/146 [==============================] - 89s 556ms/step - loss: 183.0325 - accuracy: 0.5559 - val_loss: 26.0210 - val_accuracy: 0.4069
# Epoch 2/50
# 146/146 [==============================] - 81s 556ms/step - loss: 343.5966 - accuracy: 0.5226 - val_loss: 174.4445 - val_accuracy: 0.5888
# Epoch 3/50
# 146/146 [==============================] - 81s 556ms/step - loss: 281.0430 - accuracy: 0.5448 - val_loss: 468.0154 - val_accuracy: 0.5923
# Epoch 4/50
# 146/146 [==============================] - 79s 542ms/step - loss: 913.5151 - accuracy: 0.5085 - val_loss: 598.4979 - val_accuracy: 0.4644
# Epoch 5/50
# 146/146 [==============================] - 80s 552ms/step - loss: 580.1447 - accuracy: 0.4814 - val_loss: 473.3206 - val_accuracy: 0.4635
# Epoch 6/50
# 146/146 [==============================] - 82s 565ms/step - loss: 439.1011 - accuracy: 0.4793 - val_loss: 393.4965 - val_accuracy: 0.4567
# Epoch 7/50
# 146/146 [==============================] - 82s 562ms/step - loss: 330.4567 - accuracy: 0.4930 - val_loss: 295.0495 - val_accuracy: 0.4858
# Epoch 8/50
# 146/146 [==============================] - 82s 562ms/step - loss: 252.5332 - accuracy: 0.4960 - val_loss: 215.2281 - val_accuracy: 0.4807
# Epoch 9/50
# 146/146 [==============================] - 82s 565ms/step - loss: 210.4640 - accuracy: 0.5027 - val_loss: 169.9863 - val_accuracy: 0.4979
# Epoch 10/50
# 146/146 [==============================] - 82s 562ms/step - loss: 165.8346 - accuracy: 0.5164 - val_loss: 150.3443 - val_accuracy: 0.5107
# Epoch 11/50
# 146/146 [==============================] - 82s 562ms/step - loss: 163.8631 - accuracy: 0.5278 - val_loss: 140.4970 - val_accuracy: 0.5296
# Epoch 12/50
# 146/146 [==============================] - 82s 566ms/step - loss: 147.9290 - accuracy: 0.5368 - val_loss: 145.8766 - val_accuracy: 0.5700
# Epoch 13/50
# 146/146 [==============================] - 82s 560ms/step - loss: 148.7913 - accuracy: 0.5239 - val_loss: 135.7638 - val_accuracy: 0.5330
# ...
# Epoch 49/50
# 146/146 [==============================] - 82s 562ms/step - loss: 123.1453 - accuracy: 0.5287 - val_loss: 124.9598 - val_accuracy: 0.5571
# Epoch 50/50
# 146/146 [==============================] - 55s 373ms/step - loss: 123.0096 - accuracy: 0.5312 - val_loss: 124.3328 - val_accuracy: 0.5476

# %%
model_dynamic = make_model_dynamic_lr(scale=1)
model_dynamic.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Epoch 1/50
# 2024-01-11 17:36:15.174865: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
# 2024-01-11 17:36:15.255711: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] model_pruner failed: INVALID_ARGUMENT: Graph does not contain terminal node AssignAddVariableOp_10.
# 146/146 [==============================] - ETA: 0s - loss: 2115.3188 - accuracy: 0.5327
# 2024-01-11 17:36:37.239932: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
# 146/146 [==============================] - 25s 149ms/step - loss: 2115.3188 - accuracy: 0.5327 - val_loss: 684.7841 - val_accuracy: 0.4060
# Epoch 2/50
# 146/146 [==============================] - 20s 137ms/step - loss: 319.8051 - accuracy: 0.5428 - val_loss: 780.5250 - val_accuracy: 0.3957
# Epoch 3/50
# 146/146 [==============================] - 19s 133ms/step - loss: 291.6159 - accuracy: 0.5559 - val_loss: 785.6454 - val_accuracy: 0.3983
# Epoch 4/50
# 146/146 [==============================] - 19s 129ms/step - loss: 683.4666 - accuracy: 0.5188 - val_loss: 1659.7502 - val_accuracy: 0.3991
# Epoch 5/50
# 146/146 [==============================] - 19s 129ms/step - loss: 701.1747 - accuracy: 0.5254 - val_loss: 265.7771 - val_accuracy: 0.4318
# Epoch 6/50
# 146/146 [==============================] - 19s 130ms/step - loss: 170.0303 - accuracy: 0.5439 - val_loss: 126.0876 - val_accuracy: 0.5940
# Epoch 7/50
# 146/146 [==============================] - 19s 130ms/step - loss: 228.5789 - accuracy: 0.5484 - val_loss: 101.5007 - val_accuracy: 0.5571
# Epoch 8/50
# 146/146 [==============================] - 19s 129ms/step - loss: 101.5142 - accuracy: 0.5405 - val_loss: 100.8300 - val_accuracy: 0.6472
# Epoch 9/50
# 146/146 [==============================] - 19s 128ms/step - loss: 172.0713 - accuracy: 0.5602 - val_loss: 87.4072 - val_accuracy: 0.4489
# Epoch 10/50
# 146/146 [==============================] - 19s 129ms/step - loss: 204.8789 - accuracy: 0.5340 - val_loss: 401.7147 - val_accuracy: 0.6103
# Epoch 11/50
# 146/146 [==============================] - 19s 128ms/step - loss: 209.4724 - accuracy: 0.5516 - val_loss: 118.5252 - val_accuracy: 0.5760
# Epoch 12/50
# 146/146 [==============================] - 19s 129ms/step - loss: 206.2731 - accuracy: 0.5439 - val_loss: 84.1205 - val_accuracy: 0.5622
# Epoch 13/50
# 146/146 [==============================] - 19s 129ms/step - loss: 118.3361 - accuracy: 0.5658 - val_loss: 89.0638 - val_accuracy: 0.4670
# ...
# 146/146 [==============================] - 19s 127ms/step - loss: 139.2478 - accuracy: 0.5827 - val_loss: 136.4357 - val_accuracy: 0.4575
# Epoch 42/50
# 146/146 [==============================] - 18s 126ms/step - loss: 112.9380 - accuracy: 0.5842 - val_loss: 175.2344 - val_accuracy: 0.4266
# Epoch 43/50
#  97/146 [==================>...........] - ETA: 5s - loss: 117.4427 - accuracy: 0.5764

# %%



