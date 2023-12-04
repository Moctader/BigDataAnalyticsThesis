import cv2
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import glob
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score


def is_grayscale(image_path):
    image = cv2.imread(image_path)
    return len(image.shape) < 3

def extract_features(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image is grayscale
    if is_grayscale(image_path):
        # Convert grayscale image to 3 channels
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Resize the image to 50x50
    image = cv2.resize(image, (50, 50))

    # Flatten the image to use pixel values as features
    features = image.flatten()
    
    return features


def extract_features_from_directory(image_path, label):
    feature_list = []
    labels = []
    features = extract_features(image_path)
    feature_list.append(features)
    labels.append(label)
    return np.array(feature_list), labels






main_directory = '/Users/moctader/Thesis_code/output20/*'

# Get a list of all subdirectories inside main_directory
tpi_directories = glob.glob(os.path.join(main_directory))
tpi_directories



def calculate_f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1



for tpi_directory in tpi_directories:
    # Check if the directory contains ASS data
    ass_directory = os.path.join(tpi_directory, 'ASS/')
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

    non_ass_directory = os.path.join(tpi_directory, 'non-ASS/')
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
    directory_name = os.path.basename(tpi_directory)
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
