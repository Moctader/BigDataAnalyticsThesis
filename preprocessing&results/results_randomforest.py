# %%
import geopandas as gpd
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# %%
PREFIX = "/Users/moctader/Arcada/"

data_path=F"{PREFIX}/samples15.pkl"

# Read Data
df=gpd.GeoDataFrame(
    pd.read_pickle(data_path),
    geometry="geometry"
)

# %%
#combine channel and label extracted

X = np.array([np.array(row['combined_channels']) for _, row in df.iterrows()])
label = np.array(df['label'])


# %%
# Find the unique channels(arrays) form the combined channels

unique_indices=[22, 23, 12, 11, 10, 20, 19, 24,  5, 27,  0,  3,  4, 21,  9, 15, 18, 6]
#unique_arrays, unique_indices = np.unique(X, axis=-1, return_index=True)

# get only central pixel
feature = X[:, 25, 25, unique_indices]

# %%
# splitting
X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=42)



# %%
param_grid = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [1, 2, 3, 5, 10],
    'min_samples_leaf': [1, 2, 3, 4, 6, 10],
}


# %%
# Create a base model
rf = RandomForestClassifier()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Predict the labels for the test set
y_pred = grid_search.predict(X_test)

# Compute the validation accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the validation accuracy
print(f"Validation Accuracy: {accuracy:.2f}")


#Validation Accuracy: 0.73

# %%
# Compute the validation metrics
accuracy = accuracy_score(y_test, np.round(y_pred))
precision = precision_score(y_test, np.round(y_pred))
recall = recall_score(y_test, np.round(y_pred))
f1 = f1_score(y_test, np.round(y_pred))

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Print the predictions
print("Predictions:", np.round(y_pred))


# %%
y_pred_binary = (y_pred > 0.5).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred_binary)
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

# %%



