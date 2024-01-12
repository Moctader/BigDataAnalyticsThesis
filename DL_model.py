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
PREFIX = "/Users/akusok/wrkdir/Md-dataset"  # folder with files

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
feature = X[:, 21:30, 21:30, unique_indices]

# %%
# splitting
X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=42)

# %%
def make_model(scale=1.0):
    # Define the input shape
    input_shape = (9, 9, 18)

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

    # Compile the model for binary classification
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# %%

# Train the model
model = make_model()
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 3 minutes
# Epoch 1/10
# 2024-01-10 22:39:45.043704: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
# 2024-01-10 22:39:45.118674: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] model_pruner failed: INVALID_ARGUMENT: Graph does not contain terminal node AssignAddVariableOp_10.
# 146/146 [==============================] - ETA: 0s - loss: 257.1246 - accuracy: 0.5467
# 2024-01-10 22:40:07.137836: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
# 146/146 [==============================] - 25s 129ms/step - loss: 257.1246 - accuracy: 0.5467 - val_loss: 15.5154 - val_accuracy: 0.5863
# Epoch 2/10
# 146/146 [==============================] - 17s 117ms/step - loss: 20.4569 - accuracy: 0.5802 - val_loss: 39.8053 - val_accuracy: 0.6086
# Epoch 3/10
# 146/146 [==============================] - 17s 116ms/step - loss: 27.5042 - accuracy: 0.5776 - val_loss: 15.7298 - val_accuracy: 0.6163
# Epoch 4/10
# 146/146 [==============================] - 17s 117ms/step - loss: 41.1126 - accuracy: 0.5999 - val_loss: 16.4407 - val_accuracy: 0.5116
# Epoch 5/10
# 146/146 [==============================] - 17s 119ms/step - loss: 39.5471 - accuracy: 0.5752 - val_loss: 26.5153 - val_accuracy: 0.4953
# Epoch 6/10
# 146/146 [==============================] - 17s 116ms/step - loss: 35.6085 - accuracy: 0.6072 - val_loss: 30.4171 - val_accuracy: 0.4815
# Epoch 7/10
# 146/146 [==============================] - 17s 117ms/step - loss: 46.6920 - accuracy: 0.6194 - val_loss: 45.0672 - val_accuracy: 0.6644
# Epoch 8/10
# 146/146 [==============================] - 17s 117ms/step - loss: 33.5529 - accuracy: 0.6649 - val_loss: 28.6690 - val_accuracy: 0.6215
# Epoch 9/10
# 146/146 [==============================] - 17s 117ms/step - loss: 72.6684 - accuracy: 0.6521 - val_loss: 62.2569 - val_accuracy: 0.6172
# Epoch 10/10
# 146/146 [==============================] - 17s 117ms/step - loss: 32.6011 - accuracy: 0.6909 - val_loss: 35.5113 - val_accuracy: 0.6489

# %%
model = make_model(scale=0.5)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Epoch 1/10
# 2024-01-11 15:03:27.025291: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
# 2024-01-11 15:03:27.099239: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] model_pruner failed: INVALID_ARGUMENT: Graph does not contain terminal node AssignAddVariableOp_10.
# 146/146 [==============================] - ETA: 0s - loss: 147.1699 - accuracy: 0.5302
# 2024-01-11 15:03:39.224810: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
# 146/146 [==============================] - 14s 82ms/step - loss: 147.1699 - accuracy: 0.5302 - val_loss: 35.0653 - val_accuracy: 0.5365
# Epoch 2/10
# 146/146 [==============================] - 10s 68ms/step - loss: 58.9685 - accuracy: 0.5469 - val_loss: 22.8891 - val_accuracy: 0.5579
# Epoch 3/10
# 146/146 [==============================] - 9s 64ms/step - loss: 298.4220 - accuracy: 0.5351 - val_loss: 316.2995 - val_accuracy: 0.6026
# Epoch 4/10
# 146/146 [==============================] - 9s 65ms/step - loss: 104.2881 - accuracy: 0.5576 - val_loss: 103.0438 - val_accuracy: 0.6094
# Epoch 5/10
# 146/146 [==============================] - 9s 65ms/step - loss: 99.8757 - accuracy: 0.5551 - val_loss: 49.2435 - val_accuracy: 0.5863
# Epoch 6/10
# 146/146 [==============================] - 10s 66ms/step - loss: 75.6949 - accuracy: 0.5527 - val_loss: 65.4445 - val_accuracy: 0.6060
# Epoch 7/10
# 146/146 [==============================] - 10s 71ms/step - loss: 70.2241 - accuracy: 0.5729 - val_loss: 38.8172 - val_accuracy: 0.4764
# Epoch 8/10
# 146/146 [==============================] - 10s 67ms/step - loss: 35.1212 - accuracy: 0.5907 - val_loss: 49.3544 - val_accuracy: 0.5442
# Epoch 9/10
# 146/146 [==============================] - 9s 65ms/step - loss: 29.8578 - accuracy: 0.6033 - val_loss: 31.4261 - val_accuracy: 0.6197
# Epoch 10/10
# 146/146 [==============================] - 10s 69ms/step - loss: 25.1779 - accuracy: 0.6096 - val_loss: 41.7997 - val_accuracy: 0.4996

# %%
model = make_model(scale=0.3)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Epoch 1/10
# 2024-01-11 15:05:09.790641: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
# 2024-01-11 15:05:09.866759: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] model_pruner failed: INVALID_ARGUMENT: Graph does not contain terminal node AssignAddVariableOp_10.
# 146/146 [==============================] - ETA: 0s - loss: 32.1423 - accuracy: 0.5390
# 2024-01-11 15:05:46.426005: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
# 146/146 [==============================] - 46s 295ms/step - loss: 32.1423 - accuracy: 0.5390 - val_loss: 7.7855 - val_accuracy: 0.4541
# Epoch 2/10
# 146/146 [==============================] - 42s 291ms/step - loss: 39.9581 - accuracy: 0.5593 - val_loss: 95.7412 - val_accuracy: 0.5030
# Epoch 3/10
# 146/146 [==============================] - 42s 288ms/step - loss: 26.8225 - accuracy: 0.5707 - val_loss: 7.9169 - val_accuracy: 0.5322
# Epoch 4/10
# 146/146 [==============================] - 40s 275ms/step - loss: 12.7785 - accuracy: 0.5933 - val_loss: 16.5586 - val_accuracy: 0.6215
# Epoch 5/10
# 146/146 [==============================] - 41s 280ms/step - loss: 15.8791 - accuracy: 0.5802 - val_loss: 21.0627 - val_accuracy: 0.4738
# Epoch 6/10
# 146/146 [==============================] - 40s 277ms/step - loss: 93.9415 - accuracy: 0.5787 - val_loss: 18.5107 - val_accuracy: 0.4652
# Epoch 7/10
# 146/146 [==============================] - 41s 280ms/step - loss: 27.4585 - accuracy: 0.5930 - val_loss: 20.0483 - val_accuracy: 0.6026
# Epoch 8/10
# 146/146 [==============================] - 40s 277ms/step - loss: 25.6977 - accuracy: 0.6046 - val_loss: 18.7262 - val_accuracy: 0.6498
# Epoch 9/10
# 146/146 [==============================] - 41s 278ms/step - loss: 20.1677 - accuracy: 0.6188 - val_loss: 18.0474 - val_accuracy: 0.6489
# Epoch 10/10
# 146/146 [==============================] - 41s 278ms/step - loss: 19.0208 - accuracy: 0.6194 - val_loss: 20.6925 - val_accuracy: 0.6592

# %%
model = make_model(scale=0.2)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Epoch 1/10
# 2024-01-11 15:12:04.141960: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
# 2024-01-11 15:12:04.218099: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] model_pruner failed: INVALID_ARGUMENT: Graph does not contain terminal node AssignAddVariableOp_10.
# 146/146 [==============================] - ETA: 0s - loss: 31.1963 - accuracy: 0.5422
# 2024-01-11 15:12:46.310109: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
# 146/146 [==============================] - 53s 352ms/step - loss: 31.1963 - accuracy: 0.5422 - val_loss: 9.2566 - val_accuracy: 0.5708
# Epoch 2/10
# 146/146 [==============================] - 50s 344ms/step - loss: 11.3162 - accuracy: 0.5505 - val_loss: 12.4044 - val_accuracy: 0.5674
# Epoch 3/10
# 146/146 [==============================] - 51s 349ms/step - loss: 11.6622 - accuracy: 0.5553 - val_loss: 10.8174 - val_accuracy: 0.5562
# Epoch 4/10
# 146/146 [==============================] - 51s 346ms/step - loss: 12.9576 - accuracy: 0.5666 - val_loss: 11.3972 - val_accuracy: 0.5991
# Epoch 5/10
# 146/146 [==============================] - 51s 347ms/step - loss: 8.6681 - accuracy: 0.5973 - val_loss: 8.9586 - val_accuracy: 0.5365
# Epoch 6/10
# 146/146 [==============================] - 50s 345ms/step - loss: 7.7173 - accuracy: 0.6025 - val_loss: 8.7712 - val_accuracy: 0.5794
# Epoch 7/10
# 146/146 [==============================] - 51s 347ms/step - loss: 32.5763 - accuracy: 0.5842 - val_loss: 42.0348 - val_accuracy: 0.5004
# Epoch 8/10
# 146/146 [==============================] - 50s 344ms/step - loss: 21.6773 - accuracy: 0.5699 - val_loss: 15.0200 - val_accuracy: 0.5142
# Epoch 9/10
# 146/146 [==============================] - 50s 346ms/step - loss: 16.3958 - accuracy: 0.5933 - val_loss: 19.8449 - val_accuracy: 0.6077
# Epoch 10/10
# 146/146 [==============================] - 51s 348ms/step - loss: 13.5762 - accuracy: 0.6001 - val_loss: 25.4468 - val_accuracy: 0.6112

# %%
model = make_model(scale=0.1)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Epoch 1/10
# 2024-01-11 15:20:32.667544: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
# 2024-01-11 15:20:32.736555: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] model_pruner failed: INVALID_ARGUMENT: Graph does not contain terminal node AssignAddVariableOp_10.
# 146/146 [==============================] - ETA: 0s - loss: 1.9458 - accuracy: 0.5179
# 2024-01-11 15:20:58.954346: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
# 146/146 [==============================] - 33s 217ms/step - loss: 1.9458 - accuracy: 0.5179 - val_loss: 1.3962 - val_accuracy: 0.5588
# Epoch 2/10
# 146/146 [==============================] - 33s 226ms/step - loss: 1.2400 - accuracy: 0.6079 - val_loss: 1.3352 - val_accuracy: 0.5614
# Epoch 3/10
# 146/146 [==============================] - 34s 235ms/step - loss: 1.1307 - accuracy: 0.6368 - val_loss: 1.6630 - val_accuracy: 0.5545
# Epoch 4/10
# 146/146 [==============================] - 30s 205ms/step - loss: 1.1111 - accuracy: 0.6613 - val_loss: 2.1240 - val_accuracy: 0.5854
# Epoch 5/10
# 146/146 [==============================] - 34s 236ms/step - loss: 1.2611 - accuracy: 0.6755 - val_loss: 2.2918 - val_accuracy: 0.5854
# Epoch 6/10
# 146/146 [==============================] - 33s 228ms/step - loss: 1.1618 - accuracy: 0.7062 - val_loss: 2.7818 - val_accuracy: 0.5519
# Epoch 7/10
# 146/146 [==============================] - 31s 210ms/step - loss: 2.2501 - accuracy: 0.6823 - val_loss: 2.6765 - val_accuracy: 0.5751
# Epoch 8/10
# 146/146 [==============================] - 34s 235ms/step - loss: 1.2715 - accuracy: 0.7257 - val_loss: 3.4785 - val_accuracy: 0.5983
# Epoch 9/10
# 146/146 [==============================] - 31s 211ms/step - loss: 2.0188 - accuracy: 0.7165 - val_loss: 3.5155 - val_accuracy: 0.5648
# Epoch 10/10
# 146/146 [==============================] - 33s 224ms/step - loss: 2.3472 - accuracy: 0.7079 - val_loss: 4.6364 - val_accuracy: 0.5863

# %%
model = make_model(scale=0.07)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Epoch 1/10
# 2024-01-11 15:25:59.774159: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
# 2024-01-11 15:25:59.842050: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] model_pruner failed: INVALID_ARGUMENT: Graph does not contain terminal node AssignAddVariableOp_10.
# 146/146 [==============================] - ETA: 0s - loss: 29.0844 - accuracy: 0.5274
# 2024-01-11 15:26:20.748121: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
# 146/146 [==============================] - 26s 168ms/step - loss: 29.0844 - accuracy: 0.5274 - val_loss: 25.0141 - val_accuracy: 0.5888
# Epoch 2/10
# 146/146 [==============================] - 21s 146ms/step - loss: 14.1725 - accuracy: 0.5351 - val_loss: 14.4869 - val_accuracy: 0.4515
# Epoch 3/10
# 146/146 [==============================] - 20s 139ms/step - loss: 11.6800 - accuracy: 0.5570 - val_loss: 12.6741 - val_accuracy: 0.5845
# Epoch 4/10
# 146/146 [==============================] - 21s 145ms/step - loss: 9.6055 - accuracy: 0.5720 - val_loss: 10.3753 - val_accuracy: 0.4833
# Epoch 5/10
# 146/146 [==============================] - 24s 166ms/step - loss: 8.2779 - accuracy: 0.5924 - val_loss: 9.6439 - val_accuracy: 0.6043
# Epoch 6/10
# 146/146 [==============================] - 22s 149ms/step - loss: 7.1701 - accuracy: 0.6042 - val_loss: 9.5551 - val_accuracy: 0.6180
# Epoch 7/10
# 146/146 [==============================] - 20s 138ms/step - loss: 6.1970 - accuracy: 0.6184 - val_loss: 8.7016 - val_accuracy: 0.6120
# Epoch 8/10
# 146/146 [==============================] - 22s 153ms/step - loss: 6.0520 - accuracy: 0.6250 - val_loss: 13.2620 - val_accuracy: 0.4549
# Epoch 9/10
# 146/146 [==============================] - 23s 159ms/step - loss: 6.2164 - accuracy: 0.6282 - val_loss: 7.9603 - val_accuracy: 0.5116
# Epoch 10/10
# 146/146 [==============================] - 21s 147ms/step - loss: 4.8952 - accuracy: 0.6388 - val_loss: 9.4490 - val_accuracy: 0.4850

# %%
model = make_model(scale=1.5)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Epoch 1/10
# 2024-01-11 15:29:42.254393: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
# 2024-01-11 15:29:42.322982: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] model_pruner failed: INVALID_ARGUMENT: Graph does not contain terminal node AssignAddVariableOp_10.
# 146/146 [==============================] - ETA: 0s - loss: 527.8526 - accuracy: 0.5520
# 2024-01-11 15:30:13.380533: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
# 146/146 [==============================] - 34s 224ms/step - loss: 527.8526 - accuracy: 0.5520 - val_loss: 10.6859 - val_accuracy: 0.5906
# Epoch 2/10
# 146/146 [==============================] - 31s 216ms/step - loss: 50.4616 - accuracy: 0.5830 - val_loss: 7.7673 - val_accuracy: 0.5124
# Epoch 3/10
# 146/146 [==============================] - 31s 216ms/step - loss: 19.8610 - accuracy: 0.6098 - val_loss: 9.0434 - val_accuracy: 0.4910
# Epoch 4/10
# 146/146 [==============================] - 31s 214ms/step - loss: 9.9327 - accuracy: 0.6377 - val_loss: 7.2645 - val_accuracy: 0.6455
# Epoch 5/10
# 146/146 [==============================] - 31s 215ms/step - loss: 4.5641 - accuracy: 0.7358 - val_loss: 11.8809 - val_accuracy: 0.4790
# Epoch 6/10
# 146/146 [==============================] - 31s 215ms/step - loss: 4.4930 - accuracy: 0.7607 - val_loss: 6.0700 - val_accuracy: 0.6652
# Epoch 7/10
# 146/146 [==============================] - 31s 215ms/step - loss: 2.6358 - accuracy: 0.8324 - val_loss: 6.1431 - val_accuracy: 0.5519
# Epoch 8/10
# 146/146 [==============================] - 31s 215ms/step - loss: 3.0459 - accuracy: 0.8356 - val_loss: 17.2265 - val_accuracy: 0.4532
# Epoch 9/10
# 146/146 [==============================] - 31s 215ms/step - loss: 3.1711 - accuracy: 0.8251 - val_loss: 10.4712 - val_accuracy: 0.6678
# Epoch 10/10
# 146/146 [==============================] - 31s 215ms/step - loss: 2.1700 - accuracy: 0.8641 - val_loss: 5.7713 - val_accuracy: 0.6086

# %%
model = make_model(scale=2)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Epoch 1/10
# 2024-01-11 15:35:00.380970: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
# 2024-01-11 15:35:00.457474: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] model_pruner failed: INVALID_ARGUMENT: Graph does not contain terminal node AssignAddVariableOp_10.
# 146/146 [==============================] - ETA: 0s - loss: 352.8024 - accuracy: 0.5428
# 2024-01-11 15:35:50.874437: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
# 146/146 [==============================] - 55s 361ms/step - loss: 352.8024 - accuracy: 0.5428 - val_loss: 25.2425 - val_accuracy: 0.6240
# Epoch 2/10
# 146/146 [==============================] - 51s 351ms/step - loss: 17.2524 - accuracy: 0.6188 - val_loss: 16.0336 - val_accuracy: 0.6318
# Epoch 3/10
# 146/146 [==============================] - 51s 351ms/step - loss: 15.1602 - accuracy: 0.6617 - val_loss: 13.2001 - val_accuracy: 0.6275
# Epoch 4/10
# 146/146 [==============================] - 51s 350ms/step - loss: 9.1610 - accuracy: 0.7246 - val_loss: 20.7290 - val_accuracy: 0.5330
# Epoch 5/10
# 146/146 [==============================] - 51s 352ms/step - loss: 8.5238 - accuracy: 0.7577 - val_loss: 17.3201 - val_accuracy: 0.6378
# Epoch 6/10
# 146/146 [==============================] - 51s 351ms/step - loss: 10.3660 - accuracy: 0.7682 - val_loss: 14.6229 - val_accuracy: 0.5373
# Epoch 7/10
# 146/146 [==============================] - 51s 350ms/step - loss: 6.4932 - accuracy: 0.8045 - val_loss: 20.7936 - val_accuracy: 0.4867
# Epoch 8/10
# 146/146 [==============================] - 51s 350ms/step - loss: 3.6999 - accuracy: 0.8375 - val_loss: 13.4551 - val_accuracy: 0.5768
# Epoch 9/10
# 146/146 [==============================] - 51s 351ms/step - loss: 2.7091 - accuracy: 0.8579 - val_loss: 17.0405 - val_accuracy: 0.6687
# Epoch 10/10
# 146/146 [==============================] - 51s 351ms/step - loss: 7.2167 - accuracy: 0.8105 - val_loss: 19.5657 - val_accuracy: 0.6129

# %%
model = make_model(scale=3)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# %% [markdown]
# ## better models

# %%
model = make_model(scale=0.3)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Epoch 1/50
# 2024-01-11 16:23:13.958203: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
# 2024-01-11 16:23:14.036619: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] model_pruner failed: INVALID_ARGUMENT: Graph does not contain terminal node AssignAddVariableOp_10.
# 146/146 [==============================] - ETA: 0s - loss: 103.1658 - accuracy: 0.5287
# 2024-01-11 16:23:55.576066: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
# 146/146 [==============================] - 54s 361ms/step - loss: 103.1658 - accuracy: 0.5287 - val_loss: 86.3153 - val_accuracy: 0.6189
# Epoch 2/50
# 146/146 [==============================] - 78s 536ms/step - loss: 28.8346 - accuracy: 0.5739 - val_loss: 20.7591 - val_accuracy: 0.6094
# Epoch 3/50
# 146/146 [==============================] - 81s 559ms/step - loss: 12.8078 - accuracy: 0.5677 - val_loss: 25.6021 - val_accuracy: 0.4575
# Epoch 4/50
# 146/146 [==============================] - 79s 542ms/step - loss: 27.3055 - accuracy: 0.5514 - val_loss: 15.3378 - val_accuracy: 0.4695
# Epoch 5/50
# 146/146 [==============================] - 79s 540ms/step - loss: 14.6772 - accuracy: 0.5561 - val_loss: 11.0904 - val_accuracy: 0.5811
# Epoch 6/50
# 146/146 [==============================] - 82s 560ms/step - loss: 12.9619 - accuracy: 0.5759 - val_loss: 14.2739 - val_accuracy: 0.5133
# Epoch 7/50
# 146/146 [==============================] - 82s 564ms/step - loss: 12.3207 - accuracy: 0.5804 - val_loss: 11.1525 - val_accuracy: 0.6060
# Epoch 8/50
# 146/146 [==============================] - 82s 562ms/step - loss: 8.8865 - accuracy: 0.6018 - val_loss: 11.1119 - val_accuracy: 0.4738
# Epoch 9/50
# 146/146 [==============================] - 82s 564ms/step - loss: 22.6410 - accuracy: 0.5920 - val_loss: 9.2586 - val_accuracy: 0.5107
# Epoch 10/50
# 146/146 [==============================] - 82s 563ms/step - loss: 8.5404 - accuracy: 0.6109 - val_loss: 10.2835 - val_accuracy: 0.6180
# Epoch 11/50
# 146/146 [==============================] - 82s 562ms/step - loss: 7.9388 - accuracy: 0.6250 - val_loss: 8.1945 - val_accuracy: 0.6146
# Epoch 12/50
# 146/146 [==============================] - 82s 560ms/step - loss: 6.8958 - accuracy: 0.6336 - val_loss: 8.4967 - val_accuracy: 0.5322
# Epoch 13/50
# 146/146 [==============================] - 82s 564ms/step - loss: 7.0623 - accuracy: 0.6454 - val_loss: 9.8383 - val_accuracy: 0.4893
# ...
# Epoch 49/50
# 146/146 [==============================] - 82s 561ms/step - loss: 16.4301 - accuracy: 0.6536 - val_loss: 20.5740 - val_accuracy: 0.5150
# Epoch 50/50
# 146/146 [==============================] - 82s 561ms/step - loss: 11.3605 - accuracy: 0.6971 - val_loss: 17.0023 - val_accuracy: 0.5536

# %% [markdown]
# 


