# %% [markdown]
# <a href="https://colab.research.google.com/github/Moctader/BigDataAnalyticsThesis/blob/proposed-fix/feature_vector.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
pip install mxnet==1.5.1


# %%
!apt-get install -y git


# %%
!git clone https://github.com/miaow1988/SqueezeNet_v1.2.git


# %%
ASS_folder = "/content/extracted_dataset/output20/*/ASS/"
non_ASS_folder = "/content/extracted_dataset/output20/*/non-ASS/"

# %%
from google.colab import drive
import zipfile
import os

# Mount Google Drive
drive.mount('/content/drive')

# Specify the path to the ZIP file in your Google Drive
zip_file_path = '/content/drive/MyDrive/Dataset.zip'

# Specify the directory where you want to extract the contents
extract_path = '/content/extracted_dataset/'

# Create the target directory if it doesn't exist
os.makedirs(extract_path, exist_ok=True)

# Extract the contents of the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print(f"Contents of {zip_file_path} extracted to {extract_path}")


# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
# define a simple data batch
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])
def get_image(url, show=False):
    if url.startswith('http'):
        # download and show the image
        fname = mx.test_utils.download(url)
    else:
        fname = url
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    if img is None:
         return None
    if show:
         plt.imshow(img)
         plt.axis('off')
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (224, 224))

    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    img = img / 255.0

    return img

def get_features(img):
    fe_mod.forward(Batch([mx.nd.array(img)]))
    features = fe_mod.get_outputs()[0].asnumpy()
    return features

# %%
from google.colab import files
import mxnet as mx
import os


sym, arg_params, aux_params = mx.model.load_checkpoint('/content/SqueezeNet_v1.2/model', 0)

# Create a module and bind it
mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])

# Load the parameters

# Get a list of all layers
all_layers = sym.get_internals()

# Print the list of layers
print("List of layers:")
print(all_layers.list_outputs()[-10:])


fe_sym = all_layers['flatten0_output']
fe_mod = mx.mod.Module(symbol=fe_sym, context=mx.cpu(), label_names=None)
fe_mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
fe_mod.set_params(arg_params, aux_params)



import glob
import os
# Directory containing acid sulfate soil images
ASS_folders = glob.glob(ASS_folder)
Ass_images = []

for folder in ASS_folders:
    images_in_folder = glob.glob(os.path.join(folder, '*.png'))
    Ass_images.extend(images_in_folder)

non_ASS_folders = glob.glob(non_ASS_folder)
non_Ass_images = []

for folder in non_ASS_folders:
    images_in_folder = glob.glob(os.path.join(folder, '*.png'))
    non_Ass_images.extend(images_in_folder)

all_images = Ass_images + non_Ass_images
image_paths=all_images



from os.path import isfile, join

features = []
labels = []

for img_path in all_images:
      # Check if the path is a file and ends with '.png' (case-insensitive)
      if isfile(img_path) and img_path.lower().endswith('.png'):
          # Assuming you have a function called get_features
          feature = get_features(get_image(img_path)).ravel()
          features.append(feature)
          if '/ASS/' in img_path:
            label = 1
          elif '/non-ASS/' in img_path:
              label = 0

          labels.append(label)


# %%
import pandas as pd
data = pd.DataFrame(features)
data['label'] = labels

# Save the DataFrame to a CSV file
csv_filename = '/content/feature_vector.csv'
data.to_csv(csv_filename, index=False)

#print(f"Feature vector saved to {csv_filename}")


# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


# %%
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
# Define Random Forest model
random_forest = RandomForestClassifier()

# Define hyperparameters for GridSearch
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
}

# Perform GridSearchCV for hyperparameter optimization
grid_search = GridSearchCV(random_forest, param_grid, cv=5)
start_time = time.time()

grid_search.fit(X_train, y_train)
end_time = time.time()

# Get the best hyperparameters
best_params = grid_search.best_params_

# Train Random Forest model with the best hyperparameters
rf = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth']).fit(X_train, y_train)

# Evaluate the model
y_pred=rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Report accuracy and computational time during training
#print(f"Best hyperparameters: {best_params}")
print(f"Test set accuracy: {accuracy:.2f}")


# %%



