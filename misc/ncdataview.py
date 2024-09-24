import xarray as xr
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load the NetCDF file (you can swap this out to display other types of data like precipitation, temperature, etc. Just change the file name)
file_path = os.path.dirname(os.path.realpath(__file__)) + '\\humidity.nc' # Ensure the data file is in the same folder as this script
ds = xr.open_dataset(file_path)

# Select the humidity variable, assuming 'hu' is the correct variable name
humidity = ds['hu']

# Filter the data to only include 2023
humidity_2023 = humidity.sel(time=slice('2023-01-01', '2023-12-31'))

# Subset the data to focus on Denmark (latitude: 54-58, longitude: 7-13)
humidity_2023_denmark = humidity_2023.sel(latitude=slice(54, 58), longitude=slice(7, 13))

# Calculate the average humidity for the year 2023
average_humidity_2023 = humidity_2023_denmark.mean(dim='time')

# Plotting
plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

# Plot average humidity data with pcolormesh or contourf
average_humidity_2023.plot(
    ax=ax,
    cmap='coolwarm',
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'label': 'Average Humidity (2023)'}
)

# Add geographical features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Add gridlines and labels
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}

# Set extent to zoom in on Denmark
ax.set_extent([7, 13, 54, 58], crs=ccrs.PlateCarree())

# Title
ax.set_title('Average Humidity in Denmark (2023)')

# Display the plot
plt.show()
