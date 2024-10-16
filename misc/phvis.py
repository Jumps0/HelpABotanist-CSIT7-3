import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import cm

# Load the CSV file containing the grid positions and pH values
grid_df = pd.read_csv('denmarkgrid.csv')

# Check for the pH_CaCl2 column
if 'pH_CaCl2' not in grid_df.columns:
    raise ValueError("'pH_CaCl2' column not found in the CSV file")

# Create a plot with cartopy's PlateCarree projection
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})

# Add features to the map (coastlines, borders, etc.)
ax.coastlines(resolution='10m')  # Higher resolution for more detail
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.OCEAN)

# Get the min and max pH values for normalization
min_ph = grid_df['pH_CaCl2'].min()
max_ph = grid_df['pH_CaCl2'].max()

# Normalize pH values to a range between 0 and 1 for color mapping
norm = plt.Normalize(min_ph, max_ph)
cmap = cm.get_cmap('coolwarm')  # Coolwarm gradient from blue to red

# Loop through the grid squares and add a rectangle for each, colored by pH value
for index, row in grid_df.iterrows():
    lat = row['latitude']
    lon = row['longitude']
    ph_value = row['pH_CaCl2']
    
    # Normalize the pH value and map it to the color
    color = cmap(norm(ph_value))
    
    # Create a rectangle for each grid square (0.1x0.1 degrees)
    rect = patches.Rectangle((lon, lat), 0.1, 0.1, linewidth=1,
                             edgecolor='black', facecolor=color,
                             transform=ccrs.PlateCarree())  # Ensure rectangles use the correct projection
    
    # Add the rectangle to the plot
    ax.add_patch(rect)

# Set plot limits based on the latitude and longitude ranges (for Denmark region)
ax.set_extent([grid_df['longitude'].min() - 0.5, grid_df['longitude'].max() + 1.0 - 0.5, 
               grid_df['latitude'].min() - 0.5, grid_df['latitude'].max() + 1.0 - 0.5], 
              crs=ccrs.PlateCarree())

# Add gridlines with labels for latitude and longitude
gridlines = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linewidth=1, color='gray', alpha=0.5, linestyle='--')
gridlines.top_labels = False  # Disable top labels
gridlines.right_labels = False  # Disable right labels
gridlines.xlabel_style = {'size': 12, 'color': 'black'}  # Customize the x-axis label style
gridlines.ylabel_style = {'size': 12, 'color': 'black'}  # Customize the y-axis label style

# Add a colorbar to represent the pH values
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, aspect=40)
cbar.set_label('pH (CaCl2)', fontsize=12)

# Show the plot
plt.gca().set_aspect('auto', adjustable='box')
plt.show()
