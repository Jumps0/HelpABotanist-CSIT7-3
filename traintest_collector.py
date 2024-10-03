# This script is used to gather up training and testing data for training the ML model.
# It will pick a random square on the grid, and then gather up its neighbors in a 10x10 (total) grid.
# These values are considered to be the *testing* data.
# Anything not within that area is the *training* data.
# New folders and scripts will be created to store this sectioned off data should they not exist already.

import pandas as pd
import os

# Load the grid data from 'denmarkgrid.csv'
df = pd.read_csv('denmarkgrid.csv')

# Set the flags:
require_positive_occurrence = True  # If TRUE, the central square must have at least one instance of positive occurrence.
random_test_split = True  # If TRUE, 20% of data will be selected randomly for testing. If FALSE, a "chunk" of the data will be selected uniformly in a 10x10 square.

# Create train and test directories if they don't exist
if not os.path.exists('train'):
    os.makedirs('train')
if not os.path.exists('test'):
    os.makedirs('test')

if random_test_split:
    # Randomly sample 20% of the data for testing
    test_data = df.sample(frac=0.2, random_state=42)
    # Select the remaining squares as training data
    train_data = df[~df.index.isin(test_data.index)]
else:
    # Select the central square
    if require_positive_occurrence:
        # Select only squares with positive occurrences
        positive_squares = df[df['positiveOccurences'] > 0]

        if not positive_squares.empty:
            # Randomly select a square from the ones with positive occurrences
            random_square = positive_squares.sample(n=1).iloc[0]
        else:
            raise ValueError("No grid squares with positive occurrences available.")
    else:
        # Randomly select any grid square
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

### VISUALIZATION (Optional) ###
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Visualization flag
visualize = True

if visualize:
    # Create a plot with cartopy's PlateCarree projection
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})

    # Add features to the map (coastlines, borders, etc.)
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)

    # Plot the training data in light blue
    for index, row in train_data.iterrows():
        rect = patches.Rectangle((row['longitude'], row['latitude']), 0.1, 0.1,
                                 linewidth=1, edgecolor='black', facecolor='lightblue',
                                 transform=ccrs.PlateCarree())
        ax.add_patch(rect)

    # Plot the testing data in yellow
    for index, row in test_data.iterrows():
        rect = patches.Rectangle((row['longitude'], row['latitude']), 0.1, 0.1,
                                 linewidth=1, edgecolor='black', facecolor='yellow',
                                 transform=ccrs.PlateCarree())
        ax.add_patch(rect)

    # Set plot limits based on the latitude and longitude ranges
    ax.set_extent([df['longitude'].min() - 0.5, df['longitude'].max() + 1.0 - 0.5, 
                   df['latitude'].min() - 0.5, df['latitude'].max() + 1.0 - 0.5], 
                  crs=ccrs.PlateCarree())

    # Add gridlines with labels for latitude and longitude
    gridlines = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {'size': 12, 'color': 'black'}
    gridlines.ylabel_style = {'size': 12, 'color': 'black'}

    # Title
    plt.title('Training Data (Blue) vs Testing Data (Yellow)')

    # Show the plot with an equal aspect ratio
    plt.gca().set_aspect('auto', adjustable='box')
    plt.show()
