# %%
# Import Section

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
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
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# %%
# Paths 
PREFIX = "/Users/moctader/Arcada/"

Read_data=F"{PREFIX}/samples15.pkl"

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

unique_arrays, unique_indices = np.unique(X, axis=-1, return_index=True)
#unique_indices=[22, 23, 12, 11, 10, 20, 19, 24,  5, 27,  0,  3,  4, 21,  9, 15, 18, 6]

# %%
features = X[:, :, :, unique_indices]


# %%
# Assaign other features lattitude and longitude 

#features=unique_arrays
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
from scipy import ndimage
# Define the CNN model for processing image features
input_feature = Input(shape=(50, 50, 18))

# Convolutional layers with increasing filters, dropout, batch normalization, and max pooling
x = Conv2D(8, (3, 3), activation='relu')(input_feature)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)  # Add MaxPooling
x = Dropout(0.3)(x)

x = Conv2D(16, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)  # Add MaxPooling
x = Dropout(0.3)(x)

x = Conv2D(32, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)  # Add MaxPooling
x = Dropout(0.3)(x)

x = Conv2D(64, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)  # Add MaxPooling
x = Dropout(0.3)(x)


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

# Use the Adam optimizer with a learning rate of 0.001
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Compile the model
model.compile(optimizer=custom_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    [X_feature_train, X_scalar_train],
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.1, 
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
 
)

# Evaluate the model on the test set
loss, accuracy = model.evaluate([X_feature_test, X_scalar_test], y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# Plot learning curve
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Learning Curve')
plt.ylim(0.5, 0.75)  
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# %%
y_pred = model.predict([X_feature_test, X_scalar_test])

# Convert probabilities to binary predictions
y_pred_binary = (y_pred > 0.5).astype(int)

precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
conf_matrix = confusion_matrix(y_test, y_pred_binary)

print("Metrics on Testing Data:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)


# %%
# Draw the heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.show()

# %%
# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_binary)
roc_auc = roc_auc_score(y_test, y_pred_binary)

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# %%


# %%



