# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, BatchNormalization, DepthwiseConv2D
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Activation, Add, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, roc_auc_score


# %%
# Paths 
PREFIX = "/Users/moctader/Arcada/"

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
unique_indices=[22, 23, 12, 11, 10, 20, 19, 24,  5, 27,  0,  3,  4, 21,  9, 15, 18, 6]
features = X[:, 1:49, 1:49, unique_indices]


# %%
# Train test split

X_train, X_test, y_train, y_test = train_test_split(
    features,
    label,
    test_size=0.2,
    random_state=42
)


# %%
input_layer = Input(shape=(48, 48, 18))
model = Conv2D(32, 3, strides=2, padding='same')(input_layer)
model = BatchNormalization()(model)
model = Activation('relu')(model)

def depthwise_separable_block(x, filters, strides=1):
    x = DepthwiseConv2D(3, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, 1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)  # Adjust dropout rate as needed

    return x

# Depthwise separable blocks with Dropout
model = depthwise_separable_block(model, 64)
model = depthwise_separable_block(model, 128, strides=2)  # Downsample
model = depthwise_separable_block(model, 128)
model = depthwise_separable_block(model, 256, strides=2)  # Downsample

# Global average pooling and dense layer for classification
model = GlobalAveragePooling2D()(model)
model = Dense(256, activation='relu')(model)

# Apply Dropout to the dense layer
model = Dropout(0.5)(model)  # Adjust the dropout rate as needed

output_layer = Dense(1, activation='sigmoid')(model)

# Custom learning rate for Adam optimizer
custom_optimizer = Adam(learning_rate=0.007)  # Adjust the learning rate as needed

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model with custom optimizer
model.compile(optimizer=custom_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
#model.summary()


# %%
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.ylim(0.5, 0.75)  
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# %%
#  prediction
y_pred = model.predict(X_test)

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


