import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# 1. Load train and test datasets
train_data = pd.read_csv('train/train_data.csv')
test_data = pd.read_csv('test/test_data.csv')

# 2. Select spatial features (latitude and longitude) and labels (already present)
features_train = train_data[['latitude', 'longitude']]
labels_train = train_data['label']

features_test = test_data[['latitude', 'longitude']]
labels_test = test_data['label']

# 3. Scale the spatial features
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

# 4. Create and fit the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(features_train_scaled, labels_train)

# 5. Predict on the test set
y_pred = knn.predict(features_test_scaled)

# 6. Evaluate the model
accuracy = accuracy_score(labels_test, y_pred)
report = classification_report(labels_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# 7. Hyperparameter tuning with Grid Search to find the best number of neighbors (k)
param_grid = {'n_neighbors': range(1, 21)}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(features_train_scaled, labels_train)

print(f'Best k: {grid.best_params_["n_neighbors"]}')

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Compute confusion matrix
cm = confusion_matrix(labels_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

import numpy as np

# Get the mean test scores for each value of 'k' from GridSearchCV
k_values = range(1, 21)
mean_scores = grid.cv_results_['mean_test_score']


