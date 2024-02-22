# %%
import geopandas as gpd
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from hpelm import ELM
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


# %%
PREFIX = "/Users/moctader/Arcada/"

data_path=F"{PREFIX}/samples.pkl"

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
#feature = X[:, unique_indices]
# get only central pixel
feature = X[:, 25, 25, unique_indices]

# %%
# splitting
X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=42)



# %%
# Reshape y_train to (number_of_samples, output_size)
y_train_reshaped = y_train.reshape(-1, 1)

# Initialize ELM model
elm = ELM(X_train.shape[1], y_train_reshaped.shape[1])

# Add 20 neurons with sigmoid activation
elm.add_neurons(20, "sigm")

# Add 10 neurons with radial basis function (RBF) activation and L2 norm
elm.add_neurons(10, "rbf_l2")

# Train the ELM model with Leave-One-Out (LOO) cross-validation
elm.train(X_train, y_train_reshaped, "LOO")

# Make predictions on the test data
Y_pred = elm.predict(X_test)

# Compute the validation accuracy
accuracy = accuracy_score(y_test, np.round(Y_pred))
print(f"Validation Accuracy: {accuracy:.2f}")




# %%
# Compute the validation metrics
accuracy = accuracy_score(y_test, np.round(Y_pred))
precision = precision_score(y_test, np.round(Y_pred))
recall = recall_score(y_test, np.round(Y_pred))
f1 = f1_score(y_test, np.round(Y_pred))

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")




# %%
y_pred_binary = (Y_pred > 0.5).astype(int)
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
#


