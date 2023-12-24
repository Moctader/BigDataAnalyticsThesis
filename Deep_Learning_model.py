#pip install --upgrade tensorflow


from google.colab import drive
import zipfile
import os     
import numpy as np
import cv2
import glob
import os
import cv2
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


"""
  
# Mount Google Drive
drive.mount('/content/drive')

# Specify the path to the ZIP file in your Google Drive
zip_file_path = '/content/drive/MyDrive/landsat_TPI.zip'

# Specify the directory where you want to extract the contents
extract_path = '/content/extracted_dataset/'

# Create the target directory if it doesn't exist
os.makedirs(extract_path, exist_ok=True)

# Extract the contents of the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print(f"Contents of {zip_file_path} extracted to {extract_path}")


"""

# Load images and labels
def load_images_and_labels(dataset_path, label):
    images = []
    labels = []

    if os.path.isfile(dataset_path):
        # If dataset_path is a file, treat it as a single dataset
        img_path = dataset_path

        # Load image using OpenCV
        img = cv2.imread(img_path)

        # Check if the image is not None
        if img is not None:
            img = img / 255.0
            images.append(img)
            labels.append(label)
    else:
        # If dataset_path is a directory, iterate over its contents
        for filename in os.listdir(dataset_path):
            img_path = os.path.join(dataset_path, filename)

            # Load image using OpenCV
            img = cv2.imread(img_path)

            # Check if the image is not None
            if img is not None:  
                img = img / 255.0
                images.append(img)
                labels.append(label)

    return np.asarray(images), np.asarray(labels)




# Ass
# List of folders or files containing different datasets
extracted_path = '/content/extracted_dataset/landsat_TPI/*/ASS'

dataset_paths = glob.glob(os.path.join(extracted_path, '*'))

# Load images and labels from each dataset folder or file
Assimages = []
Asslabels = []

for idx, dataset_path in enumerate(dataset_paths):
    label = 1  # Assign a unique label for each dataset
    images, labels = load_images_and_labels(dataset_path, label)

    Assimages.extend(images)
    Asslabels.extend(labels)

# Convert lists to numpy arrays
Assimages = np.asarray(Assimages)
Asslabels = np.asarray(Asslabels)




# Non-ASS
# List of folders or files containing different datasets
non_Ass_extracted_path = '/content/extracted_dataset/landsat_TPI/*/non-ASS'

dataset_paths = glob.glob(os.path.join(non_Ass_extracted_path, '*'))

# Load images and labels from each dataset folder or file
non_Ass_images = []
non_Ass_labels = []

for idx, dataset_path in enumerate(dataset_paths):
    label = 0  # Assign a unique label for each dataset
    images, labels = load_images_and_labels(dataset_path, label)

    non_Ass_images.extend(images)
    non_Ass_labels.extend(labels)

# Convert lists to numpy arrays
non_Ass_images = np.asarray(non_Ass_images)
non_Ass_labels = np.asarray(non_Ass_labels)


#features and labels

features = np.r_[Assimages, non_Ass_images]
labels = np.r_[Asslabels, non_Ass_labels]

# Split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)



# Create an instance of the data generator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.1
)

# Model build

class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.Constant(0.01), bias_initializer=tf.keras.initializers.Constant(0.1))
        self.dense2 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.Constant(0.02), bias_initializer=tf.keras.initializers.Constant(0.2))
        self.dense3 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.Constant(0.03), bias_initializer=tf.keras.initializers.Constant(0.3))
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense4 = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.Constant(0.04), bias_initializer=tf.keras.initializers.Constant(0.4))
        self.dense5 = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.Constant(0.05), bias_initializer=tf.keras.initializers.Constant(0.5))

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dropout(x)
        x = self.dense4(x)
        return self.dense5(x)


# Create an instance of the custom model
model = CustomModel()

# Use tf.data.Dataset for better performance
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(32)

# Assuming X_val, y_val for the validation set
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(32)

# Define the loss and optimizer
loss_object = tf.keras.losses.BinaryCrossentropy()

initial_learning_rate = 0.01
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

# Define accuracy metrics
train_accuracy = tf.keras.metrics.BinaryAccuracy()
val_accuracy = tf.keras.metrics.BinaryAccuracy()

# Initialize variables for learning rate adjustment
best_val_loss = float('inf')  # Set to positive infinity initially
best_epoch = 0


# Training step
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_accuracy.update_state(labels, predictions)
    return loss

# Validation step
def val_step(images, labels):
    predictions = model(images)
    val_accuracy.update_state(labels, predictions)
    

epochs = 100
for epoch in range(epochs):
    for images, labels in train_dataset:
        # Convert tensors to numpy arrays, apply data augmentation, and convert back to tensors
        images = np.array(images)
        for i in range(len(images)):
            images[i] = datagen.random_transform(images[i])
        images = tf.convert_to_tensor(images)
        loss = train_step(images, labels)

    # Calculate accuracy on the validation set
    for val_images, val_labels in val_dataset:
        val_step(val_images, val_labels)

    # Get accuracy values
    train_acc_value = train_accuracy.result().numpy()
    val_acc_value = val_accuracy.result().numpy()

    print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}, Train Accuracy: {train_acc_value}, Val Accuracy: {val_acc_value}')



