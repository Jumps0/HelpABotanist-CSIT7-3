import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load the CSV file containing the grid positions
grid_df = pd.read_csv('denmarkgrid.csv')

# Create a plot with cartopy's PlateCarree projection
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})

# Add features to the map (coastlines, borders, etc.)
ax.coastlines(resolution='10m')  # Higher resolution for more detail
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.OCEAN)

# Function to generate a random color
def random_color():
    return np.random.rand(3,)

# Loop through the grid squares and add a rectangle for each
for index, row in grid_df.iterrows():
    lat = row['latitude'] - 0.05  # Shift down by 0.05 degrees
    lon = row['longitude'] - 0.05  # Shift left by 0.05 degrees
    
    # Create a rectangle for each grid square (0.1x0.1 degrees)
    rect = patches.Rectangle((lon, lat), 0.1, 0.1, linewidth=1,
                             edgecolor='black', facecolor=random_color(),
                             transform=ccrs.PlateCarree())  # Ensure rectangles use the correct projection
    
    # Add the rectangle to the plot
    ax.add_patch(rect)

# Set plot limits based on the latitude and longitude ranges (for Denmark region) -- the -0.5 nonesense is to make sure the window is the right side
ax.set_extent([grid_df['longitude'].min() - 0.5, grid_df['longitude'].max() + 1.0 - 0.5, 
               grid_df['latitude'].min() - 0.5, grid_df['latitude'].max() + 1.0 - 0.5], 
              crs=ccrs.PlateCarree())

# Add gridlines with labels for latitude and longitude
gridlines = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linewidth=1, color='gray', alpha=0.5, linestyle='--')
gridlines.top_labels = False  # Disable top labels
gridlines.right_labels = False  # Disable right labels
gridlines.xlabel_style = {'size': 12, 'color': 'black'}  # Customize the x-axis label style
gridlines.ylabel_style = {'size': 12, 'color': 'black'}  # Customize the y-axis label style

# Show the plot with an equal aspect ratio
plt.gca().set_aspect('auto', adjustable='box')
plt.show()
