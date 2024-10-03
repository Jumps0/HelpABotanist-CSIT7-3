import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load the updated grid data
grid_file = "denmarkgrid.csv"
grid_df = pd.read_csv(grid_file)

# Define color maps for positive and negative occurrences
positive_colormap = mcolors.LinearSegmentedColormap.from_list("green_white", ["green", "white"], N=100)
negative_colormap = mcolors.LinearSegmentedColormap.from_list("white_red", ["white", "red"], N=100)

# Extract latitudes, longitudes, and occurrences
latitudes = grid_df['latitude']
longitudes = grid_df['longitude']
positive_occurrences = grid_df['positiveOccurences']
negative_occurrences = grid_df['negativeOccurences']

# Determine the size of the grid based on unique lat/lon values
lat_range = np.unique(latitudes)
lon_range = np.unique(longitudes)

# Create the plot with a map of Denmark in the background
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

# Add background map of Denmark
ax.set_extent([7, 13, 54, 58])  # Longitude and latitude bounds for Denmark
ax.coastlines(resolution='10m')
ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=.5)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.gridlines(draw_labels=True)

# Loop through each grid square and draw a rectangle based on the occurrence data
for i, row in grid_df.iterrows():
    lat = row['latitude']
    lon = row['longitude']
    pos_count = row['positiveOccurences']
    neg_count = row['negativeOccurences']

    # Calculate the difference between positive and negative occurrences
    occurrence_diff = pos_count - neg_count

    # Determine the color based on whether there are more positives or negatives
    if occurrence_diff > 0:
        # More positive occurrences, map to green gradient
        color_value = min(occurrence_diff / 100, 1.0)  # Normalize to max 100 occurrences
        color = positive_colormap(color_value)
    else:
        # More negative occurrences, map to red gradient
        color_value = min(abs(occurrence_diff) / 100, 1.0)  # Normalize to max 100 occurrences
        color = negative_colormap(color_value)

    # Draw the rectangle (using a 0.1 x 0.1 degree grid)
    rect = plt.Rectangle((lon, lat), 0.1, 0.1, color=color, transform=ccrs.PlateCarree())
    ax.add_patch(rect)

# Set labels and title
ax.set_title("Flower Occurrence Heat Map over Denmark")

# Add a color bar to show the gradient scale
sm = plt.cm.ScalarMappable(cmap=positive_colormap, norm=plt.Normalize(vmin=-100, vmax=100))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.7)
cbar.set_label('Occurrence Difference (Positive - Negative)')

# Display the heat map
plt.show()
