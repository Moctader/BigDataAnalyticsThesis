import zipfile
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import cv2
import numpy as np
from scipy.stats import skew, kurtosis
import glob
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score


def extract_features_from_directory(image_path, label):
    feature_list = []
    labels = []
    features = extract_central_pixel(image_path)
    feature_list.append(features)
    labels.append(label)
    #print(feature_list)
    #print(labels)

    return np.array(feature_list), labels


def extract_central_pixel(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculate the coordinates of the center pixel
    height, width = image.shape
    center_row = int(np.floor(height / 2))
    center_col = int(np.floor(width / 2))

    # Extract the central pixel value
    central_pixel = image[center_row, center_col, None]

    return central_pixel

def calculate_moments(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 

    # Flatten the pixel values into a 1D array
    pixel_values = image.flatten()

    # Calculate statistical moments
    mean_value = np.mean(pixel_values)
    variance_value = np.var(pixel_values)
    skewness_value = skew(pixel_values)
    kurtosis_value = kurtosis(pixel_values)
    
    
    # # Calculate GLCM
    # distances = [1, 2, 3]  # List of pixel pair distances
    # angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # List of pixel pair angles

    # glcm = greycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)

    # Extract relevant statistics from GLCM
    # glcm_mean = np.mean(glcm)
    # glcm_variance = np.var(glcm)
    # glcm_skewness = skew(glcm.flatten())
    # glcm_kurtosis = kurtosis(glcm.flatten())

    return mean_value, variance_value, skewness_value, kurtosis_value



def calculate_f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1



main_directory = '/Users/moctader/Thesis_code/output20/*'


# Get a list of all subdirectories inside main_directory
directories = glob.glob(os.path.join(main_directory))


for directory in directories:
    # Check if the directory contains ASS data
    ass_directory = os.path.join(directory, 'ASS/')
    dataset_paths = glob.glob(os.path.join(ass_directory, '*'))

    # Load images and labels from the ASS directory
    Assimages = []
    Asslabels = []

    for idx, dataset_path in enumerate(dataset_paths):
        label = 1  # Assign a unique label for each dataset
        images, labels = extract_features_from_directory(dataset_path, label)

        # Append to the existing lists
        Assimages.extend(images)
        Asslabels.extend(labels)

    # Convert lists to numpy arrays
    Assimages = np.asarray(Assimages)
    Asslabels = np.asarray(Asslabels)

    # Load images and labels from the non-ASS directory
    non_Ass_images = []
    non_Ass_labels = []

    non_ass_directory = os.path.join(directory, 'non-ASS/')
    dataset_paths = glob.glob(os.path.join(non_ass_directory, '*'))

    for idx, dataset_path in enumerate(dataset_paths):
        label = 0  # Assign a unique label for each dataset
        images, labels = extract_features_from_directory(dataset_path, label=0)

        # Append to the existing lists
        non_Ass_images.extend(images)
        non_Ass_labels.extend(labels)

    # Convert lists to numpy arrays
    non_Ass_images = np.asarray(non_Ass_images)
    non_Ass_labels = np.asarray(non_Ass_labels)

    # Combine ASS and non-ASS data
    data = np.r_[Assimages, non_Ass_images]
    targets = np.r_[Asslabels, non_Ass_labels]

    nan_mask = np.isnan(data)

    # Calculate mean along the columns (axis=0)
    column_means = np.nanmean(data, axis=0)

    # Fill NaN values with the mean of each column
    data = np.where(nan_mask, column_means, data)


    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, targets, random_state=42)

    # Define Random Forest model
    random_forest = RandomForestClassifier()
    start_time_rf = time.time()

    rf = random_forest.fit(X_train, y_train)

    end_time_rf = time.time()

    # Evaluate the Random Forest model
    y_pred_rf = rf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)

    # Report results for the current TPI directory with Random Forest
    directory_name = os.path.basename(directory)
    print(f"\nResults for Random Forest in {directory_name}:")
    precision_rf, recall_rf, f1_rf = calculate_f1_score(y_test, y_pred_rf)
    print(f'Precision: {precision_rf:.2f}')
    print(f'Recall: {recall_rf:.2f}')
    print(f'F1 Score: {f1_rf:.2f}')
    print(f"Test set accuracy: {accuracy_rf:.2f}")
    print(f"Computational time: {end_time_rf - start_time_rf:.2f} seconds\n")

    # Define Logistic Regression model

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logistic_regression = LogisticRegression(max_iter=1000, solver='liblinear')
    start_time_lr=time.time()
    lr = logistic_regression.fit(X_train_scaled, y_train)
    end_time_lr=time.time()
    y_pred_lr = lr.predict(X_test_scaled)
    accuracy_lr = accuracy_score(y_test, y_pred_lr)


    # Report results for Logistic Regression in the current TPI directory
    print(f"\nResults for Logistic Regression in {directory_name}:")
    precision_lr, recall_lr, f1_lr = calculate_f1_score(y_test, y_pred_lr)
    print(f'Precision: {precision_lr:.2f}')
    print(f'Recall: {recall_lr:.2f}')
    print(f'F1 Score: {f1_lr:.2f}')
    print(f"Test set accuracy: {accuracy_lr:.2f}")
    print(f"Computational time: {end_time_lr - start_time_lr:.2f} seconds\n")

    # Compare results between Random Forest and Logistic Regression
    print(f"Comparison between Random Forest and Logistic Regression in {directory_name}:")
    print(f"Random Forest F1 Score: {f1_rf:.2f}")
    print(f"Logistic Regression F1 Score: {f1_lr:.2f}")
