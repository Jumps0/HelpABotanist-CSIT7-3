import pandas as pd
import networkx as nx
import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load the dataset (swap out with 1 2 or 3 if you wish)
data = pd.read_csv("datagrid_1.csv")

# Step 2: Build the graph with nodes and edges
G = nx.Graph()

# Add each data point as a node
for idx, row in data.iterrows():
    lat, lon = row['latitude'], row['longitude']
    pos_occurrence = row['positiveOccurrence']
    node = (lat, lon)
    # Initialize node with data as attributes
    G.add_node(node, 
               tg=row['tg'], tx=row['tx'], tn=row['tn'],
               hu=row['hu'], rr=row['rr'],
               soilType=row['soilType'], pH=row['pH_CaCl2'],
               soil_moisture=row['soil_moisture'],
               positiveOccurrence=pos_occurrence)

# Add edges based on spatial proximity (adjacent cells)
for node in G.nodes:
    lat, lon = node
    neighbors = [
        (lat + 0.1, lon), (lat - 0.1, lon), (lat, lon + 0.1), (lat, lon - 0.1),
        (lat + 0.1, lon + 0.1), (lat + 0.1, lon - 0.1), (lat - 0.1, lon + 0.1), (lat - 0.1, lon - 0.1)
    ]
    for neighbor in neighbors:
        if neighbor in G.nodes:
            G.add_edge(node, neighbor)

# Step 3: Initialize labels for Label Propagation with binary occurrence
# Create a dictionary to store initial labels (1 for positive, 0 for negative)
labels = {}
for node, data in G.nodes(data=True):
    labels[node] = 1 if data['positiveOccurrence'] > 0 else 0

# Prepare data for Label Propagation
X = np.array(list(G.nodes))
y = np.array([labels[node] for node in G.nodes])

# Split nodes for training and testing (keep 80% for training)
train_indices, test_indices = train_test_split(range(len(X)), test_size=0.2, stratify=y)
y_train = np.copy(y)
y_train[test_indices] = -1  # Mark test data as unlabeled for label propagation

# Step 4: Fit the Label Propagation model
label_prop_model = LabelPropagation(max_iter=1000, kernel='rbf') # Use 'rbf' or 'knn'. Though RBF is better
label_prop_model.fit(X, y_train)

# Retrieve the predictions for all nodes
predicted_labels = label_prop_model.transduction_

# Step 5: Calculate and print accuracy
# Only evaluate on test indices, where we know the actual label
y_test_true = y[test_indices]
y_test_pred = predicted_labels[test_indices]

accuracy = accuracy_score(y_test_true, y_test_pred)
print(f"Accuracy on test set: {accuracy * 100:.2f}%")

import seaborn as sns
import matplotlib.pyplot as plt

# Confusion Matrix Visualization with Labels
conf_matrix = confusion_matrix(y_test_true, y_test_pred)
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
plt.title('Confusion Matrix [Label Propogation]')
plt.show()

# Optionally, print predicted labels for verification
print_all_nodes = False
if print_all_nodes:
    for i, node in enumerate(G.nodes):
        G.nodes[node]['predicted_label'] = predicted_labels[i]
        if i in test_indices:
            print(f"Node {node}: Predicted Label: {predicted_labels[i]}, Actual Label: {y[i]}")
