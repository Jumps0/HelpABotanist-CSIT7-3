import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def load_and_preprocess_data(filepath, plant_column):
    # Load and preprocess data for a specific plant column.

    data = pd.read_csv(filepath)
    
   # print(f"Columns in the dataset: {data.columns}")
   # print(f"First few rows:\n{data.head()}")  # Print the first few rows

    data['label'] = data[plant_column].apply(lambda x: 1 if x > 0 else 0)
    
    print( data['label'],"= data['label']")
    # Features and target
    X = data.drop(columns=['longitude', 'latitude', 'label', 'isTest', 'predicted', 'soilType'] + [plant_column])
    y = data['label']
    
   
    # Scale numerical features
    scaler = StandardScaler()
    data['original_latitude'] = data['latitude']
    data['original_longitude'] = data['longitude']
    X_scaled = scaler.fit_transform(X)
    
    
    # Split train-test based on isTest column
    X_train = X_scaled[data['isTest'] == False]
    y_train = y[data['isTest'] == False]
    X_test = X_scaled[data['isTest'] == True]
    y_test = y[data['isTest'] == True]
    
    
    return X_train, X_test, y_train, y_test, data

def update_with_pred(data, y_pred, filepath):
    # Update the CSV file with the 'predicted' column.

    # Check if "predicted" column exists; if not, insert it after "isTest"
    if "predicted" not in data.columns:
        predicted_index = data.columns.get_loc("isTest") + 1
        data.insert(predicted_index, "predicted", np.nan)
    
    # Update only rows where isTest == True
    test_indices = data[data['isTest'] == True].index
    data.loc[test_indices, "predicted"] = y_pred

    # Save the updated file
    data.to_csv(filepath, index=False)
    print(f"CSV file updated with predictions and saved to: {filepath}")

def train_and_evaluate_knn(X_train, X_test, y_train, y_test, n_neighbors=5, detailed=False, update_csv=False, path=None, data=None):
    # Train a KNN Classifier and evaluate its performance.

    # Initialize KNN model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    cv_scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy')
    
    # Train the model
    knn.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = knn.predict(X_test)
    
    # Print results
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    if detailed: # Split off advanced results, only show if requested
        print(f"Cross-Validation Accuracy Scores: {cv_scores}")
        print(f"Mean Cross-Validation Accuracy: {np.mean(cv_scores):.4f}")
        print(f"Standard Deviation of Cross-Validation Accuracy: {np.std(cv_scores):.4f}")
        
        print("\nTest Set Evaluation:")
        print("Classification Report:\n", classification_report(y_test, y_pred))

        plot_confusion_matrix(y_test, y_pred)
    
    if update_csv and path:
        update_with_pred(data, y_pred, path)
        print("Complete train_and_evaluate_knn")
       
    return accuracy

def plot_confusion_matrix(y_true, y_pred):
    # Plot a confusion matrix with annotations.

    conf_matrix = confusion_matrix(y_true, y_pred)
    labels = [
        f'TN\n{conf_matrix[0, 0]}',  # True Negative
        f'FP\n{conf_matrix[0, 1]}',  # False Positive
        f'FN\n{conf_matrix[1, 0]}',  # False Negative
        f'TP\n{conf_matrix[1, 1]}'   # True Positive
    ]
    labels = np.array(labels).reshape(2, 2)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='Blues',
                xticklabels=['Predicted: No Occurrence', 'Predicted: Occurrence'],
                yticklabels=['Actual: No Occurrence', 'Actual: Occurrence'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix [KNN]')
    plt.show()

def run_for_all_plants(filepath, plant_start_index, n_neighbors=5):
    # Run the KNN model for all plant columns starting at a specified index.

    data = pd.read_csv(filepath)
    plant_columns = data.columns[plant_start_index:]
    
    print(f"Found {len(plant_columns)} plants to process.\n")
    
    all_accuracy = []

    # Go through all plants
    for plant in plant_columns:
        #print("="*60)
        #print(f"Processing Plant: {plant}")
        try:
            acc = main(filepath, plant, n_neighbors)
            all_accuracy.append(acc)
        except Exception as e:
            print(f"Error processing plant {plant}: {e}")
        #print("="*60)
    
    print(f'FINAL AVERAGE ACCURACY: {sum(all_accuracy) / len(all_accuracy)}')

def run_for_all_plants_detailed(filepath, plant_start_index, output_txt="plant_resultsKNN.txt"):

    # Run the GB model for all plant columns starting at a specified index, save results to a .txt file.
    
    # Parameters:
    # - filepath: Path to the input CSV file.
    # - plant_start_index: Column index where plant data starts.
    # - output_txt: Path to save the accuracy results as a text file.

    import os

    # Load the data
    data = pd.read_csv(filepath)
    plant_columns = data.columns[plant_start_index:]
    
    print(f"Found {len(plant_columns)} plants to process.\n")
    
    all_accuracy = []
    plant_results = []

    # Open the results file for writing
    with open(output_txt, "w") as f:
        f.write("Plant Name | Total Occurrences | Accuracy\n")
        f.write("=" * 50 + "\n")

        # Go through all plants
        for plant in plant_columns:
            try:
                # Count total occurrences
                total_occurrences = data[plant].sum()
                
                # Run the main function to get accuracy
                acc = main(filepath, plant, n_neighbors=7)
                all_accuracy.append(acc)
                
                # Save the results
                result_line = f"{plant} | {total_occurrences} | {acc:.4f}\n"
                plant_results.append((plant, acc))
                f.write(result_line)

            except Exception as e:
                error_line = f"Error processing plant {plant}: {e}\n"
                print(error_line)
                f.write(error_line)
        
        # Calculate overall metrics
        if all_accuracy:
            avg_accuracy = sum(all_accuracy) / len(all_accuracy)
            min_accuracy = min(all_accuracy)
            max_accuracy = max(all_accuracy)

            # Write final metrics
            f.write("\n")
            f.write("=" * 50 + "\n")
            f.write(f"FINAL AVERAGE ACCURACY: {avg_accuracy:.4f}\n")
            f.write(f"LOWEST ACCURACY: {min_accuracy:.4f}\n")
            f.write(f"HIGHEST ACCURACY: {max_accuracy:.4f}\n")
        else:
            f.write("\nNo plants processed successfully.\n")

    print(f"Results have been saved to {os.path.abspath(output_txt)}")


def main(filepath, plant_column, n_neighbors=5):
    X_train, X_test, y_train, y_test, data = load_and_preprocess_data(filepath, plant_column)
    print(f"Running model for plant column: {plant_column} with {n_neighbors} neighbors")
    return train_and_evaluate_knn(X_train, X_test, y_train, y_test, n_neighbors, path=filepath, detailed=False, update_csv=True, data=data)
    
""" # Uncomment this out if you want to run it solo
if __name__ == "__main__":
    data_file = "datagrid.csv"
    plant_column = "Andromeda polifolia"
    main(data_file, plant_column)
    #run_for_all_plants(data_file, 12)
    #run_for_all_plants_detailed(data_file, 12)
"""

# NOTE:
# AVERAGE OVERALL ACCURACY: 78.61%
# RANGE: 62-92%