
import os
from glob import glob
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# Function to extract central pixel and its coordinates from an image
def extract_central_pixel_and_coords(image_path):
    img = Image.open(image_path)
    width, height = img.size
    center_x = width // 2
    center_y = height // 2
    central_pixel = img.getpixel((center_x, center_y))
    return central_pixel, center_x, center_y

# Directory containing acid sulfate soil images
ASS_folder = "/Users/moctader/Thesis_code/output20/*/Ass/"
ASS_folders = glob(ASS_folder)
Ass_images = []

for folder in ASS_folders:
    images_in_folder = glob(os.path.join(folder, '*.png'))
    Ass_images.extend(images_in_folder)

non_ASS_folder = "/Users/moctader/Thesis_code/output20/*/Non-Ass/"
non_ASS_folders = glob(non_ASS_folder)
non_Ass_images = []

for folder in non_ASS_folders:
    images_in_folder = glob(os.path.join(folder, '*.png'))
    non_Ass_images.extend(images_in_folder)

all_images = Ass_images + non_Ass_images
print(all_images)
features = []
labels = []
for img_path in all_images:
    central_pixel, center_x, center_y = extract_central_pixel_and_coords(img_path)
    features.append((central_pixel[0], central_pixel[1], central_pixel[2]))  # Flatten pixel values

    if "/Ass/" in img_path:
        labels.append(1)
    elif "/Non-Ass/" in img_path:
        labels.append(0)      

X = np.array(features)
y = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],  
    'max_depth': [None, 10, 20, 30],  
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 2, 4],  
}

# Create a RandomForestClassifier
rf_model = RandomForestClassifier()

# Create the GridSearchCV object
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='f1_weighted')

# Fit the model to the data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Use the best model for predictions
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')  # Calculate F1 score

print("Accuracy:", accuracy)
print("F1 Score:", f1)