from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from label import labelpropagation, build_graph,results
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split
from knn_complete import main
from rf_complete import main
from gb_complete import main

# Initialize FastAPI
app = FastAPI()

# Allow all origins (you can restrict this to specific origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (use a list of allowed origins for better security)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Path to the CSV file
CSV_FILE_PATH = "datagrid.csv"

# Pydantic model for input data
class PlantData(BaseModel):
    plant: str  # User-selected plant name
   

# Function to get plant details from the CSV
def get_plant_details_from_csv(file_path: str, plant: str):
    # Read CSV and use the first row as header
    plant_data = pd.read_csv(file_path, header=0)
    #print("CSV Data:", plant_data.head())  # Print the first few rows of the CSV to debug

    # Check if the plant exists in the columns
    if plant not in plant_data.columns:
        raise HTTPException(status_code=404, detail=f"Plant '{plant}' not found in the file.")

    # Get the row corresponding to the plant's data
    plant_details_row = plant_data[plant_data[plant] == 1].iloc[0]  # Get the first match
   # print(f"Plant Details Row: {plant_details_row}")  # Print plant details for debugging

    # Map the columns to the dictionary
    plant_details_dict = {
        'latitude': plant_details_row['latitude'],
        'longitude': plant_details_row['longitude'],
        'tg': plant_details_row['tg'],
        'tx': plant_details_row['tx'],
        'tn': plant_details_row['tn'],
        'hu': plant_details_row['hu'],
        'rr': plant_details_row['rr'],
        'soilType': plant_details_row['soilType'],
        'pH_CaCl2': plant_details_row['pH_CaCl2'],
        'soil_moisture': plant_details_row['soil_moisture'],
        'isTest': plant_details_row['isTest'],
        'predicted': plant_details_row['predicted'] if 'predicted' in plant_details_row else "Unknown"  # Check if 'predicted' exists
    }
    print("get_plant_details_from_csv RUN SUCCESSFULLY !!")
   # print("Mapped Plant Details:", plant_details_dict)  # Debug: Final details dictionary
    return plant_details_dict


# Define theendpoint
@app.post("/predict_LP")
async def predict_location(data: PlantData):
    try:
        # Extract the plant name from the request
        target_plant = data.plant
        print(f"Target plant received: {target_plant}")

        # Retrieve plant details (latitude, longitude, etc.) from the CSV
        plant_details = get_plant_details_from_csv(CSV_FILE_PATH, target_plant)
      
         # Build the graph for the plant
        graph = build_graph(target_plant)
        
        # Perform label propagation to classify plant type at each node
        print("Starting label propagation...")
        graph_finished = labelpropagation(graph, finish_labels=True, use_similarity=True, start_percentage=0.2, interations=100)
        print("Label propagation completed.")

         # Query the label propagation result
        lat = plant_details['latitude']
        lon = plant_details['longitude']
        prediction =results(graph_finished, update_file=True, file_name="datagrid.csv")

        print(prediction)
        if lat is None or lon is None:
            raise HTTPException(status_code=404, detail="Latitude or longitude not found for this plant.")
        
         # Prepare heatmap data
        heatmap = [
            {
                "latitude": node.get("original_latitude", node["latitude"]),  # Get original latitude
                "longitude": node.get("original_longitude", node["longitude"]),  # Get original longitude
                "intensity": node["label"][0] if "label" in node and isinstance(node["label"], (list, tuple)) else 0,
                "isTest": node.get("isTest", False),
                "isTrain": node.get("isTrain", False)
           
            }

            for node in graph_finished.nodes.values()
        ]
       
        
        return {"heatmap": heatmap,"prediction": prediction}  # Correctly return the heatmap data
       
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/predict_knn")
async def predict_knn(data: PlantData):
    csvfile = "datagrid2.csv"
    try:
        # Extract the plant name from input
        plant_column = data.plant
        print(f"Target plant received: {plant_column}")

        # Load the dataset
        full_data = pd.read_csv(csvfile)

        # Ensure the plant column exists
        if plant_column not in full_data.columns:
            raise HTTPException(status_code=400, detail=f"Plant column '{plant_column}' does not exist in the dataset.")

        # Split the data into train and test sets
        X = full_data[["latitude", "longitude"]]  # Features
        y = full_data[plant_column]               # Target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print('KNN Starts....')
        # Call the main function to train the model and get accuracy
        accuracy = main(csvfile, plant_column)
        print("KNN Finished.....")

        print(f"Model trained for plant '{plant_column}' with accuracy: {accuracy:.2f}")

        # Prepare heatmap data
        heatmap_data = [
            {
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "intensity": 1 if idx in y_test.index else 0,  # Mark as 1 if in the test set
                "isTest": idx in X_test.index,
                "isTrain": idx in X_train.index,
            }
            for idx, row in full_data.iterrows()
        ]

        # Return heatmap and accuracy
        return {
            "heatmap": heatmap_data,
            "prediction": f"{accuracy:.2f}",
        }

    except KeyError as ke:
        raise HTTPException(status_code=400, detail=f"Missing required column: {str(ke)}")
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=f"Data processing error: {str(ve)}")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/predict_RF")
async def predict_RF(data: PlantData):
    csvfile = "datagrid3.csv"
    try:
        # Extract the plant name from input
        plant_column = data.plant
        print(f"Target plant received: {plant_column}")

        # Load the dataset
        full_data = pd.read_csv(csvfile)

        # Ensure the plant column exists
        if plant_column not in full_data.columns:
            raise HTTPException(status_code=400, detail=f"Plant column '{plant_column}' does not exist in the dataset.")

        # Split the data into train and test sets
        X = full_data[["latitude", "longitude"]]  # Features
        y = full_data[plant_column]               # Target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print('RF Starts....')
        # Call the main function to train the model and get accuracy
        accuracy = main(csvfile, plant_column)
        print("RF Finished.....")

        print(f"Model trained for plant '{plant_column}' with accuracy: {accuracy:.2f}")

        # Prepare heatmap data
        heatmap_data = [
            {
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "intensity": 1 if idx in y_test.index else 0,  # Mark as 1 if in the test set
                "isTest": idx in X_test.index,
                "isTrain": idx in X_train.index,
            }
            for idx, row in full_data.iterrows()
        ]

        # Return heatmap and accuracy
        return {
            "heatmap": heatmap_data,
            "prediction": f"{accuracy:.2f}",
        }

    except KeyError as ke:
        raise HTTPException(status_code=400, detail=f"Missing required column: {str(ke)}")
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=f"Data processing error: {str(ve)}")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

   

@app.post("/predict_GBC")
async def predict_GBC(data: PlantData):
    csvfile = "datagrid4.csv"
    try:
        # Extract the plant name from input
        plant_column = data.plant
        print(f"Target plant received: {plant_column}")

        # Load the dataset
        full_data = pd.read_csv(csvfile)

        # Ensure the plant column exists
        if plant_column not in full_data.columns:
            raise HTTPException(status_code=400, detail=f"Plant column '{plant_column}' does not exist in the dataset.")

        # Split the data into train and test sets
        X = full_data[["latitude", "longitude"]]  # Features
        y = full_data[plant_column]               # Target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print('GBC Starts....')
        # Call the main function to train the model and get accuracy
        accuracy = main(csvfile, plant_column)
        print("GBC Finished.....")

        print(f"Model trained for plant '{plant_column}' with accuracy: {accuracy:.2f}")

        # Prepare heatmap data
        heatmap_data = [
            {
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "intensity": 1 if idx in y_test.index else 0,  # Mark as 1 if in the test set
                "isTest": idx in X_test.index,
                "isTrain": idx in X_train.index,
            }
            for idx, row in full_data.iterrows()
        ]

        # Return heatmap and accuracy
        return {
            "heatmap": heatmap_data,
            "prediction": f"{accuracy:.2f}",
        }

    except KeyError as ke:
        raise HTTPException(status_code=400, detail=f"Missing required column: {str(ke)}")
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=f"Data processing error: {str(ve)}")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
