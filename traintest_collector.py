# This script is used to gather up training and testing data for training the ML model.
# It will pick a random square on the grid, and then gather up its neighbors in a 10x10 (total) grid.
# These values are considered to be the *testing* data.
# Anything not within that area is the *training* data.
# New folders and scripts will be created to store this sectioned off data should they not exist already.

import pandas as pd
import os
import random

# Load the grid data from 'denmarkgrid.csv'
df = pd.read_csv('denmarkgrid.csv')

# Create train and test directories if they don't exist
if not os.path.exists('train'):
    os.makedirs('train')
if not os.path.exists('test'):
    os.makedirs('test')

# Randomly select one grid square
random_square = df.sample(n=1).iloc[0]
lat = random_square['latitude']
lon = random_square['longitude']

# Define the 10x10 square area around the selected square
size = 0.5
lat_range = (lat - size, lat + size)
lon_range = (lon - size, lon + size)

# Select the squares within this area (test data)
test_data = df[(df['latitude'] >= lat_range[0]) & (df['latitude'] <= lat_range[1]) &
               (df['longitude'] >= lon_range[0]) & (df['longitude'] <= lon_range[1])]

# Select the remaining squares as training data
train_data = df[~df.index.isin(test_data.index)]

# Save the test and training data to CSV files
test_file = 'test/test_data.csv'
train_file = 'train/train_data.csv'

test_data.to_csv(test_file, index=False)
train_data.to_csv(train_file, index=False)

# Print the number of squares added to each pool
print(f"{len(test_data)} squares added to the testing pool.")
print(f"{len(train_data)} squares added to the training pool.")