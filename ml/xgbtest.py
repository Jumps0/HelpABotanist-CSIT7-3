# Aided by: https://www.datacamp.com/tutorial/xgboost-in-python

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np

# Load data (datagrid_#.csv)
data = pd.read_csv("datagrid_2.csv")

# Create label based on positive occurrences
data['label'] = data['positiveOccurrence'].apply(lambda x: 1 if x > 0 else 0)

# Data preprocessing (encode soilType, scale numerical features)
data = pd.get_dummies(data, columns=['soilType'])

# Split data into features and target
X = data.drop(columns=['label', 'positiveOccurrence', 'negativeOccurrence'])
y = data['label']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize XGBoost model
xgb_model = xgb.XGBClassifier(n_estimators=250, learning_rate=0.1, max_depth=3, eval_metric='logloss')

# Perform cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True)  # 5-fold cross-validation
cv_scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='accuracy')

# Print cross-validation results
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {np.mean(cv_scores):.4f}")
print(f"Standard Deviation of Cross-Validation Accuracy: {np.std(cv_scores):.4f}")

# Train-test split for confusion matrix visualization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model and make predictions
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# Evaluate the model
print("\nAccuracy on Test Set:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Visualization with Labels
conf_matrix = confusion_matrix(y_test, y_pred)
labels = [
    f'TN\n{conf_matrix[0, 0]}',  # True Negative
    f'FP\n{conf_matrix[0, 1]}',  # False Positive
    f'FN\n{conf_matrix[1, 0]}',  # False Negative
    f'TP\n{conf_matrix[1, 1]}'   # True Positive
]
labels = np.array(labels).reshape(2, 2)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='Blues', xticklabels=['Predicted: No Occurrence', 'Predicted: Occurrence'], 
            yticklabels=['Actual: No Occurrence', 'Actual: Occurrence'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix [xGBoost]')
plt.show()