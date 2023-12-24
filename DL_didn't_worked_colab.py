from google.colab import drive
import zipfile
import os
import glob
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

# # Mount Google Drive
# drive.mount('/content/drive')
# zip_file_path = '/content/drive/MyDrive/Dataset.zip'
# extract_path = '/content/extracted_dataset/'
# os.makedirs(extract_path, exist_ok=True)

# # Extract the contents of the ZIP file
# with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#     zip_ref.extractall(extract_path)
# print(f"Contents of {zip_file_path} extracted to {extract_path}")



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
            # Normalize to [0, 1]
            img = img / 255.0

            # Resize images if needed  
            img = cv2.resize(img, (64, 64))

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
                # Normalize to [0, 1]
                img = img / 255.0

                img = cv2.resize(img, (64, 64))

                images.append(img)
                labels.append(label)

    return np.asarray(images), np.asarray(labels)



# List of folders or files containing different datasets
extracted_path = '/content/extracted_dataset/output20/*/ASS'

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





# List of folders or files containing different datasets
non_Ass_extracted_path = '/content/extracted_dataset/output20/*/non-ASS'

dataset_paths = glob.glob(os.path.join(non_Ass_extracted_path, '*'))

# Load images and labels from each dataset folder 
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


# creat feaature data and targets
data = np.r_[Assimages, non_Ass_images]
targets = np.r_[Asslabels, non_Ass_labels]
x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.20)



# Model Sequential
model = Sequential()

# Convolutional layers with max pooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten the output before fully connected layers
model.add(Flatten())

# Fully connected layers with dropout
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))



optimizer = Adam(learning_rate=0.001)  
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train,batch_size=15,epochs=100,validation_data=(x_test, y_test))