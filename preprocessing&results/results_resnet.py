# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, Dropout, BatchNormalization
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Activation, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
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
def resnet_block(model, filters, kernel_size=3, stride=1):
    # Shortcut
    shortcut = model
    
    # First convolution layer
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(model)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Apply Dropout
    x = Dropout(0.4)(x) 
    
    # Second convolution layer
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Apply Dropout
    x = Dropout(0.4)(x) 
    
    # Adjust dimensions of the shortcut to match processed output
    if model.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same')(model)
    
    # Skip connection
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

input_layer = Input(shape=(48, 48, 18))
model = Conv2D(64, 7, strides=2, padding='same')(input_layer)
model = BatchNormalization()(model)
model = Activation('relu')(model)

# Residual blocks with Dropout
model = resnet_block(model, 8)
model = resnet_block(model, 16)
model = resnet_block(model, 32, stride=2)  # Downsample
model = resnet_block(model, 64, stride=2)  # Downsample

# Flatten and dense layers for classification
model = Flatten()(model)
model = Dense(64, activation='relu')(model)

# Apply Dropout to the dense layer
model = Dropout(0.5)(model)  

output_layer = Dense(1, activation='sigmoid')(model)

model = Model(inputs=input_layer, outputs=output_layer)


# %%
# Compile the model with custom optimizer
model.compile(optimizer=Adam(learning_rate=0.0006)  , loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
#model.summary()


# Train the model and save history
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2,
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
)

# Evaluate the model on the test set
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



