import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("datagrid_2.csv")

# Create label based on positive occurrences
# Assuming presence if positiveOccurrence > 0
data['label'] = data['positiveOccurrence'].apply(lambda x: 1 if x > 0 else 0)

# Data preprocessing (encode soilType, scale numerical features)
# One-hot encoding for soilType
data = pd.get_dummies(data, columns=['soilType'])

# Split data into features and target
X = data.drop(columns=['label', 'positiveOccurrence', 'negativeOccurrence'])
y = data['label']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and train model
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['No Occurrence', 'Occurrence'], 
            yticklabels=['No Occurrence', 'Occurrence'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
plt.show()