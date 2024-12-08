import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Load the CSV file
csv_file = 'datagrid.csv'
data = pd.read_csv(csv_file)

# Extract coordinates (latitude and longitude)
latitudes = data.iloc[:, 0]
longitudes = data.iloc[:, 1]

# Extract flower occurrence data (starting at column index 10)
flower_data = data.iloc[:, 10:]

grid_size = 0.1  # Grid cells are 0.1 x 0.1 degrees

# Define color normalization with cap at 10 occurrences
color_max = 10
norm = plt.Normalize(vmin=0, vmax=color_max)

# Custom colormap: Black for 0, Coolwarm for 1-10+ occurrences
from matplotlib.colors import ListedColormap

def custom_colormap():
    # Create a colormap with black for 0 occurrences
    base_cmap = plt.cm.coolwarm  # Original gradient
    colors = base_cmap(np.linspace(0, 1, 256))
    colors[0] = [0, 0, 0, 1]  # Set the first color (for 0 occurrences) to black
    return ListedColormap(colors)

cmap = custom_colormap()

# Create a function to generate a heatmap for a single flower
def generate_heatmap(flower_column, ax):
    ax.clear()
    ax.set_title(f"Occurrences of {flower_column}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    occurrences = flower_data[flower_column].values

    ax.scatter(longitudes, latitudes, c=occurrences, cmap=cmap, s=100, marker='s', edgecolors='none', norm=norm)

    ax.set_xlim(longitudes.min() - grid_size, longitudes.max() + grid_size)
    ax.set_ylim(latitudes.min() - grid_size, latitudes.max() + grid_size)

fig, ax = plt.subplots(figsize=(10, 8))

# Update the heatmap for animation
def update(frame):
    flower_column = flower_data.columns[frame]
    generate_heatmap(flower_column, ax)

# Set up the animation
ani = animation.FuncAnimation(fig, update, frames=len(flower_data.columns), interval=1000, repeat=True)

# Show the map & animation
plt.tight_layout()
plt.show()
