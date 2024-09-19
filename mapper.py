import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# Load the provided dataset
file_path = os.path.dirname(os.path.realpath(__file__)) + '\\data.csv' # Make sure the data is in the same folder as this program
data = pd.read_csv(file_path)

# Display the first few rows to understand the structure of the data
#print(data.head())
#print(data.info())

# Identify key columns
key_columns = ['gbifID','countryCode','eventDate','decimalLatitude', 'decimalLongitude']

# Create geometry column
geometry = [Point(xy) for xy in zip(data['decimalLongitude'], data['decimalLatitude'])]

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(data, geometry=geometry)

# Load the shapefile
shapefile_path = os.path.dirname(os.path.realpath(__file__)) + '\\mapdata\\ne_110m_admin_0_countries.shp'  # Update this path to the actual path
world = gpd.read_file(shapefile_path)

# Set up the map projection
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.Mercator()})

# Add features to the map
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAKES, edgecolor='black')
ax.add_feature(cfeature.RIVERS)

# Plot the data points
gdf.plot(ax=ax, marker='o', color='red', markersize=5, transform=ccrs.PlateCarree())

# Set the extent to focus on Denmark (latitude: 54-58, longitude: 7-13)
ax.set_extent([7, 13, 54, 58], crs=ccrs.PlateCarree())

# Add gridlines with labels
gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                  linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
gl.top_labels = False  # Disable top labels
gl.right_labels = False  # Disable right-side labels
gl.xlabel_style = {'size': 10, 'color': 'black'}  # Customize x-axis label style
gl.ylabel_style = {'size': 10, 'color': 'black'}  # Customize y-axis label style

# Customize the plot
ax.set_title('Occurrences in Denmark')

# -- More info on map plotting can be found here: https://geopandas.org/en/stable/docs/user_guide/mapping.html

# Show plot
plt.show()
