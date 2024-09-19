import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
import geodatasets
import os
import numpy as np




# Load the provided dataset
file_path = os.path.dirname(os.path.realpath(__file__)) + '\occurrence.csv' # Make sure the data is in the same folder as this program
data = pd.read_csv(file_path)

# Display the first few rows to understand the structure of the data
print(data.head())
print(data.info())  # This will show you details about columns and missing values


# This will show you how many missing values there are in each of the key columns
print("\nMissing values per column:")
key_columns = ['gbifID','countryCode','eventDate','decimalLatitude', 'decimalLongitude']
missing_data = data[key_columns].isnull().sum()
print(missing_data)


# We NEED TO DROP the null lines




# Drop rows with missing values in the key columns
key_columns = ['gbifID', 'countryCode', 'eventDate', 'decimalLatitude', 'decimalLongitude']
data_cleaned = data.dropna(subset=key_columns)


# Check how many rows remain after dropping
print(f"\nRemaining rows after dropping missing values: {len(data_cleaned)}")

import matplotlib.pyplot as plt
plt.scatter(x=data_cleaned['decimalLongitude'], y=data_cleaned['decimalLatitude'])
plt.show()

exit()