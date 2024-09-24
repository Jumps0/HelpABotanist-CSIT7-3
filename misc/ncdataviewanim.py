import xarray as xr
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation

### NOTE GRID SIZE is 11.1km (latitude) by 7.39 km (longitude)

# Load the NetCDF file
file_path = os.path.dirname(os.path.realpath(__file__)) + '\\humidity.nc' # Ensure the data file is in the same folder as this script
ds = xr.open_dataset(file_path)

# Select the humidity variable, assuming 'hu' is the correct variable name
humidity = ds['hu']

# Filter the data to only include 2023
humidity_2023 = humidity.sel(time=slice('2023-01-01', '2023-12-31'))

# Subset the data to focus on Denmark (latitude: 54-58, longitude: 7-13)
humidity_2023_denmark = humidity_2023.sel(latitude=slice(54, 58), longitude=slice(7, 13))

# Prepare the plot
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}
ax.set_extent([7, 13, 54, 58], crs=ccrs.PlateCarree())

# Initialize the plot with the first day's data
img = humidity_2023_denmark.isel(time=0).plot(
    ax=ax,
    cmap='coolwarm',
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'label': 'Humidity'},
    #add_colorbar=False,
)

# Animation function to update the plot for each day
def update(frame):
    ax.clear()  # Clear the axis to update the plot
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--').top_labels = False
    ax.set_extent([7, 13, 54, 58], crs=ccrs.PlateCarree())
    img = humidity_2023_denmark.isel(time=frame).plot(
        ax=ax,
        cmap='coolwarm',
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
    )
    ax.set_title(f'Humidity in Denmark on {str(humidity_2023_denmark.time[frame].values)[:10]}')

# Set the total duration of the animation
total_frames = len(humidity_2023_denmark.time)
interval = 60000 / total_frames  # One minute duration for the entire animation

# Create the animation
ani = FuncAnimation(fig, update, frames=total_frames, interval=interval)

# Display the animation
plt.show()
