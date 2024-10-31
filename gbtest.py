import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("datagrid_2.csv")

# Create label based on positive occurrences
data['label'] = data['positiveOccurrence'].apply(lambda x: 1 if x > 0 else 0)

# Data preprocessing (encode soilType, scale numerical features)
data = pd.get_dummies(data, columns=['soilType'])

# Define features and target
X = data.drop(columns=['label', 'positiveOccurrence', 'negativeOccurrence'])
y = data['label']

# Compute class weights
class_weights = class_weight.compute_sample_weight('balanced', y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and train Gradient Boosting model with sample weights
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gb.fit(X_train, y_train, sample_weight=class_weights[:len(y_train)])

# Make predictions
y_pred = gb.predict(X_test)

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

### NOTES (Testing with different values. Largely inconclusive) ###
# n_estimators: 100 | learning_rate: 0.1 | max_depths: 3 --- 79% 86% 85% (Default)
# n_estimators: 150 | learning_rate: 0.1 | max_depths: 3 --- 82% 80% 83% (Avg Better?)
# n_estimators: 250 | learning_rate: 0.1 | max_depths: 3 --- 81% 89% 85% (Avg Better?)
# n_estimators: 500 | learning_rate: 0.1 | max_depths: 3 --- 83% 77% 76% (Avg Worse?)
# n_estimators: 100 | learning_rate: 0.2 | max_depths: 3 --- 81% 79% 78% (Avg Worse?)
# n_estimators: 100 | learning_rate: 0.5 | max_depths: 3 --- 82% 78% 78% (Avg Worse?)
# n_estimators: 100 | learning_rate: 0.1 | max_depths: 5 --- 80% 83% 83% (Avg)
# n_estimators: 100 | learning_rate: 0.1 | max_depths: 10 -- 73% 78% 75% (Avg Worse)
# n_estimators: 100 | learning_rate: 0.1 | max_depths: 15 -- 83% 82% 76% (Avg Worse?)
# n_estimators: 250 | learning_rate: 0.2 | max_depths: 5 --- 79% 84% 81% (Avg Worse?)