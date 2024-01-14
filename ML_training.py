# %%
import geopandas as gpd
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# %%
# path

data_path="/Users/moctader/Thesis_code/pickle/samples.pkl"
data_path = "/Users/akusok/wrkdir/Golam/samples.pkl"

# %%
# Read Data
df=gpd.GeoDataFrame(
    pd.read_pickle(data_path),
    geometry="geometry"
)

# %%
# select Feature and label
X = np.array([np.array(row['combined_channels']) for _, row in df.iterrows()])
label = np.array(df['label'])

# get only central pixel
X = X[:, 25, 25, :]

# %%
# select only the unique features
# unique_arrays, unique_indices = np.unique(X, axis=-1, return_index=True)
# feature=unique_arrays

# same thing but without waiting
unique_indices = [12, 10, 11, 20, 22, 23, 19,  4, 27, 24,  0,  9, 21,  3,  5, 15, 18, 6]
feature = X[:, unique_indices]

# %%
# splitting
X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=42)

# %%
# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [1, 2, 3, 5, 10],
    'min_samples_leaf': [1, 2, 3, 4, 6, 10],
}

# Create a base model
rf = RandomForestClassifier()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

print(f"Best parameters: {best_params}")

# Best parameters: {'max_depth': 20, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 300}

# %%
# Predict the labels for the test set
y_pred = grid_search.predict(X_test)

# Compute the validation accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the validation accuracy
print(f"Validation Accuracy: {accuracy}")

# Validation Accuracy: 0.7081545064377682

# %%



