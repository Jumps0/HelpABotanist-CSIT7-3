import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the NetCDF file
file_path = os.path.dirname(os.path.realpath(__file__)) + '\\topsoildata\\moisture\\moisture_month (4).nc' # Ensure the data file is in the same folder as this script
ds = xr.open_dataset(file_path)

# Extract the soil moisture data
soil_moisture = ds['sm'].isel(time=0)  # Select the first time slice

# Create a plot
plt.figure(figsize=(12, 6))
# Use the 'imshow' method for 2D data
img = plt.imshow(soil_moisture, extent=[ds.lon.min(), ds.lon.max(), ds.lat.min(), ds.lat.max()],
                 origin='lower', cmap='viridis', interpolation='nearest')

# Add a colorbar
cbar = plt.colorbar(img, orientation='vertical')
cbar.set_label('Soil Moisture (units)', rotation=270, labelpad=15)

# Add labels and title
plt.title('Soil Moisture for a Month in 2023')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Show the plot
plt.show()
