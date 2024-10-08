### An all in one data preparation script for flower observation data ###
## What do we need to do?
# 1. Parse the .txt file into an all in one .csv file we can look at later (but won't use for ML)
# 2. Remove any observation points where we don't have exact coordinate data.
# 2b. (Optional) Add in missing regions using a web query call
# 3. Remove any duplicate entries (same location & same day)
# 4. Create a new .csv file based on a template we have (only use the first 8 columns)
# - But what if I don't have a template? Create one using weather & soil data.
# 5. Create and fill up the occurences columns
# 6. Add an empty labels column

import csv
import os
import shutil
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

current_dir = os.path.dirname(os.path.abspath(__file__)) # Absolute path
input_file_name = 'DATA (Plant 2 aka Common)\\occurrence.txt'  # Input file location (change to whatever yours is)

output_file_name = 'occurrence.csv'  # Output file location & name

### 1. Parse the initial .txt file
with open(input_file_name, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
# The first line contains the headers
headers = lines[0].strip().split('\t')

# The remaining lines contain the data
data_lines = [line.strip().split('\t') for line in lines[1:]]

# Write the output to a CSV file
with open(output_file_name, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    
    # Write the header row
    writer.writerow(headers)
    
    # Write the data rows
    writer.writerows(data_lines)

print(f"(1) Data has been parsed and saved to {output_file_name}")

# Load the .csv file (data frame) for future use
df = pd.read_csv(output_file_name)

### 1b. Limit region to Denmark
df = df[df['countryCode'] == 'DK']

### 2. Empty coordinate position removal
df = df.dropna(subset=['decimalLatitude', 'decimalLongitude'])
print(f"(2) Removed any entries missing coordinate data in {output_file_name}")

## 2b. (Optional) Region updating
do_region_update = False

if(do_region_update):
    print("(2b) Updating any missing regions... (this may take a while)")

    # Initialize the Nominatim Geocoder
    geolocator = Nominatim(user_agent="denmark_flower_observations")

    # Rate limiter to avoid hitting the query limit too quickly
    geocode = RateLimiter(geolocator.reverse, min_delay_seconds=0.15)

    # Define a function to get region and town from coordinates
    def get_region_and_town(row):
        if pd.isna(row['level1Name']) or pd.isna(row['level2Name']):
            try:
                # Perform reverse geocoding using latitude and longitude
                location = geocode((row['decimalLatitude'], row['decimalLongitude']), exactly_one=True, language='en')
                if location:
                    address = location.raw.get('address', {})
                    # Extract region and town information
                    row['level1Name'] = address.get('state', '')  # Adjust if necessary
                    row['level2Name'] = address.get('town', address.get('city', address.get('village', '')))
            except Exception as e:
                print(f"Error processing coordinates ({row['decimalLatitude']}, {row['decimalLongitude']}): {e}")
        return row

    # Apply the function to the rows with missing level1Name or level2Name
    df = df.apply(get_region_and_town, axis=1)

    print(f"(2b) Region data updated in {output_file_name}")

### 3. Duplicate observation entry removal

# Find exact duplicates based on decimalLatitude, decimalLongitude, and eventDate
duplicates = df.duplicated(subset=['decimalLatitude', 'decimalLongitude', 'eventDate'], keep='first')

# Number of duplicate rows getting removed
num_removed = duplicates.sum()

# Remove the duplicates, keeping the first occurrence
df = df[~duplicates]

print(f"(3) Removed {num_removed} duplicate entries in {output_file_name}")

# !!! Save the new occurences.csv file !!!
df.to_csv(output_file_name, index=False)
print(f"...{output_file_name} has been saved.")

### 4. Create a new .csv file specifically for ML usage

## Do we have an existing template?
template_name = 'ml_template.csv'
if(os.path.isfile(template_name)):
    # We do, continue as normal
    print(f"(4) ML Template {template_name} found successfully.")
else:
    # Uh oh, we need to cobble together a new one!
    print(f"(4) Failed to find ML Template with name {template_name}, creating new template based on weather & soil data.")

    ### STEP 1: Make a new .csv template file
    empty = pd.DataFrame(list())
    empty.to_csv(template_name)

    print("...making new template csv file")

    # Load the new .csv file (We re-use the df_grid variable later so why not)
    df_grid = pd.read_csv(template_name)

    ### STEP 2: Make the grid
    import netCDF4 as nc
    # Open the .nc files | Hope you have these, sorry about the file sizes
    # NOTE: If you don't have these files, go here: https://surfobs.climate.copernicus.eu/dataaccess/access_eobs_chunks.php
    #       and download the '0.1 deg. regular grid' for the years 2011-2024 under "Ensemble mean".
    #       Pick "TG", "TX", "TN", HU", "RR". Some of these files are big, sorry. But this is as small as they will get.
    nc_files = {
        'tg': 'weatherdata\\temperature.nc',
        'tx': 'weatherdata\\temperature_max.nc',
        'tn': 'weatherdata\\temperature_min.nc',
        'hu': 'weatherdata\\humidity.nc',
        'rr': 'weatherdata\\precipitation.nc'
    }

    print("...loading weather data")

    # First we actually need to make the grid in the empty .csv file, we can do this using one of the pre-existing .nc files
    ds = nc.Dataset('weatherdata\\precipitation.nc')

    # Extract longitude and latitude values
    longitudes = ds.variables['longitude'][:]
    latitudes = ds.variables['latitude'][:]

    # Filter the latitude and longitude values based on the specified range | Denmark so (latitude: 54-58, longitude: 7-13)
    lat_mask = (latitudes >= 54) & (latitudes <= 58)
    lon_mask = (longitudes >= 7) & (longitudes <= 13)

    # Create a meshgrid to represent the grid of latitude and longitude points
    lat_grid, lon_grid = np.meshgrid(latitudes[lat_mask], longitudes[lon_mask], indexing='ij')

    # Flatten the grids to create pairs of latitude and longitude
    lat_grid_flat = lat_grid.flatten()
    lon_grid_flat = lon_grid.flatten()

    # Create a pandas DataFrame and populate the latitude and longitude columns
    df_grid = pd.DataFrame({'latitude': lat_grid_flat, 'longitude': lon_grid_flat})

    # Shift down and left by 0.05 because of how coordinate squares are positioned
    df_grid['latitude'] = df_grid['latitude'] - 0.05
    df_grid['longitude'] = df_grid['longitude'] - 0.05

    # Finally, carefully carve out unneccessary values that aren't actually in Denmark
    df_grid = df_grid[df_grid['latitude'] >= 54.5] # Remove values below the bottom
    df_grid = df_grid[df_grid['latitude'] <= 57.8] # Remove values above the top
    df_grid = df_grid[df_grid['longitude'] < 12.5] # Remove values to the right (in Sweden)
    df_grid = df_grid[~((df_grid['latitude'] > 56.2) & (df_grid['longitude'] > 11.5))] # Remove the top right corner (Sweden)

    print("...grid created.")

    # Now get back to buisiness
    # Load data from the .nc files into dictionaries
    data_dict = {}
    for key, filename in nc_files.items():
        with nc.Dataset(filename) as dataset:
            # Assume longitude and latitude dimensions are named 'longitude' and 'latitude'
            lons = dataset.variables['longitude'][:]
            lats = dataset.variables['latitude'][:]
            
            # Get time variable and extract indices for 2023
            time_var = dataset.variables['time']
            time_units = time_var.units
            time_values = nc.num2date(time_var[:], units=time_units)
            
            # Filter time values for the year 2023
            time_2023_indices = np.where(np.array([t.year for t in time_values]) == 2023)[0]
            
            # Extract the 2023 data and compute the average for the year
            values = dataset.variables[key]
            if values.ndim == 3:  # Time, latitude, longitude
                values_2023 = values[time_2023_indices, :, :]
                values_avg_2023 = np.mean(values_2023, axis=0)  # Average over 2023
            elif values.ndim > 3:
                raise ValueError(f"Unexpected number of dimensions in {key}: {values.ndim}")
            
            data_dict[key] = {'lons': lons, 'lats': lats, 'values_avg_2023': values_avg_2023}

    # Helper function to find the nearest index for a given lat/lon
    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    # Function to get value for a specific lat/lon, retrying with +0.05N and +0.05E if needed
    def get_value_or_retry(lon, lat, data):
        lon_idx = find_nearest(data['lons'], lon)
        lat_idx = find_nearest(data['lats'], lat)
        
        value = data['values_avg_2023'][lat_idx, lon_idx]
        
        # Check if the value is masked (missing)
        if np.ma.is_masked(value):
            # Retry with +0.05 latitude and +0.05 longitude
            new_lon = lon + 0.05
            new_lat = lat + 0.05
            lon_idx_retry = find_nearest(data['lons'], new_lon)
            lat_idx_retry = find_nearest(data['lats'], new_lat)
            
            value_retry = data['values_avg_2023'][lat_idx_retry, lon_idx_retry]
            if np.ma.is_masked(value_retry):
                return np.nan  # Return NaN if both original and retry values are missing
            else:
                return float(value_retry)  # Return retry value if valid
        else:
            return float(value)  # Return original value if valid

    # Iterate through the CSV rows and add the average 2023 weather data
    for index, row in df_grid.iterrows():
        lon = row['longitude']
        lat = row['latitude']

        for key, data in data_dict.items():
            value = get_value_or_retry(lon, lat, data)
            df_grid.loc[index, key] = value
    
    # Remove any empty values (usually in the sea)
    df_grid = df_grid.dropna(subset=['rr'])
    df_grid = df_grid.dropna(subset=['hu'])

    print("...weather data loaded")

    ### 3. Add soil data
    import geopandas as gpd

    print("...loading soil data")

    shapefile_path = os.path.join(current_dir, "soildata", "ESDB (Joined Shape File)", "SGDB_PTR.shp") # NOTE: Change the file location & name if needed

    # Load the soil shapefile data
    soil_data = gpd.read_file(shapefile_path)

    # Extract major and minor parts from the 'WRBFU' column
    soil_data['WRB_major'] = soil_data['WRBFU'].str[:2]  # Major group (first two letters)
    soil_data['WRB_minor'] = soil_data['WRBFU'].str[2:]  # Minor qualifier (next two letters)

    # Dictionary for major soil groups (WRB Reference Soil Groups, see [page 153]: https://www.isric.org/sites/default/files/WRB_fourth_edition_2022-12-18.pdf)
    wrb_major_names = {
        "AC": "Acrisol", "AL": "Alisol", "AN": "Andosol", "AT": "Anthrosol",
        "AR": "Arenosol", "CL": "Calcisol", "CM": "Cambisol", "CH": "Chernozem",
        "CR": "Cryosol", "DU": "Durisol", "FR": "Ferralsol", "FL": "Fluvisol",
        "GL": "Gleysol", "GY": "Gypsisol", "HS": "Histosol", "KS": "Kastanozem",
        "LP": "Leptosol", "LX": "Lixisol", "LV": "Luvisol", "NT": "Nitisol",
        "PH": "Phaeozem", "PL": "Planosol", "PT": "Plinthosol", "PZ": "Podzol",
        "RG": "Regosol", "RT": "Retisol", "SC": "Solonchak", "SN": "Solonetz",
        "ST": "Stagnosol", "TC": "Technosol", "UM": "Umbrisol", "VR": "Vertisol"
    }

    # Dictionary for minor soil qualifiers
    wrb_minor_names = {
        'ap': 'Abruptic', 'ae': 'Aceric', 'ac': 'Acric', 'ao': 'Acroxic', 'at': 'Activic', 'ay': 'Aeolic', 'kf': 'Akrofluvic', 'km': 'Akromineralic', 'kk': 'Akroskeletic',
        'ab': 'Albic', 'ax': 'Alcalic', 'al': 'Alic', 'aa': 'Aluandic', 'an': 'Andic', 'aq': 'Anthraquic', 'ak': 'Anthric', 'am': 'Anthromollic', 'aw': 'Anthroumbric',
        'ah': 'Archaic', 'ar': 'Arenic', 'ad': 'Arenicolic', 'aj': 'Areninovic', 'as': 'Argisodic', 'ai': 'Aric', 'az': 'Arzic', 'bc': 'Biocrustic', 'br': 'Brunic',
        'by': 'Bryic', 'ca': 'Calcaric', 'cc': 'Calcic', 'cf': 'Calcifractic', 'cm': 'Cambic', 'cp': 'Capillaric', 'cb': 'Carbic', 'cn': 'Carbonatic', 'cx': 'Carbonic',
        'ch': 'Chernic', 'cq': 'Claric', 'cl': 'Chloridic', 'cr': 'Chromic', 'ce': 'Clayic', 'cj': 'Clayinovic', 'cs': 'Coarsic', 'co': 'Cohesic', 'cu': 'Columnic',
        'cd': 'Cordic', 'cy': 'Cryic', 'ct': 'Cutanic', 'dn': 'Densic', 'df': 'Differentic', 'do': 'Dolomitic', 'ds': 'Dorsic', 'dr': 'Drainic', 'du': 'Duric',
        'dy': 'Dystric', 'jk': 'Ejectiskeletic', 'ek': 'Ekranic', 'ed': 'Endic', 'et': 'Entic', 'ep': 'Epic', 'ec': 'Escalic', 'eu': 'Eutric', 'es': 'Eutrosilic',
        'ev': 'Evapocrustic', 'fl': 'Ferralic', 'fr': 'Ferric', 'fe': 'Ferritic', 'fi': 'Fibric', 'ft': 'Floatic', 'fv': 'Fluvic', 'fo': 'Folic', 'fc': 'Fractic',
        'fk': 'Fractiskeletic', 'fg': 'Fragic', 'ga': 'Garbic', 'ge': 'Gelic', 'gt': 'Gelistagnic', 'go': 'Geoabruptic', 'gr': 'Geric', 'gi': 'Gibbsic',
        'gg': 'Gilgaic', 'gc': 'Glacic', 'gl': 'Gleyic', 'gs': 'Glossic', 'gz': 'Greyzemic', 'gm': 'Grumic', 'gy': 'Gypsic', 'gf': 'Gypsofractic', 'gp': 'Gypsiric',
        'ha': 'Haplic', 'hm': 'Hemic', 'hi': 'Histic', 'ht': 'Hortic', 'hu': 'Humic', 'hg': 'Hydragric', 'hy': 'Hydric', 'hf': 'Hydrophobic', 'jl': 'Hyperalic',
        'ja': 'Hyperartefactic', 'jc': 'Hypercalcic', 'ju': 'Hyperduric', 'jd': 'Hyperdystric', 'je': 'Hypereutric', 'jf': 'Hyperferritic', 'jb': 'Hypergarbic',
        'jq': 'Hypergeric', 'jg': 'Hypergypsic', 'jh': 'Hyperhumic', 'jy': 'Hyperhydragric', 'jm': 'Hypermagnesic', 'jn': 'Hypernatric', 'jo': 'Hyperorganic',
        'jz': 'Hypersalic', 'jr': 'Hypersideralic', 'jp': 'Hyperspodic', 'jj': 'Hyperspolic', 'js': 'Hypersulfidic', 'jt': 'Hypertechnic', 'ji': 'Hyperthionic',
        'jx': 'Hyperurbic', 'ws': 'Hyposulfidic', 'wi': 'Hypothionic', 'im': 'Immissic', 'ic': 'Inclinic', 'iy': 'Inclinigleyic', 'iw': 'Inclinistagnic',
        'ia': 'Infraandic', 'is': 'Infraspodic', 'ir': 'Irragric', 'il': 'Isolatic', 'ip': 'Isopteric', 'ka': 'Kalaic', 'll': 'Lamellic', 'ld': 'Lapiadic',
        'la': 'Laxic', 'le': 'Leptic', 'lg': 'Lignic', 'lm': 'Limnic', 'ln': 'Limonic', 'lc': 'Linic', 'li': 'Lithic', 'lh': 'Litholinic', 'lx': 'Lixic',
        'lo': 'Loamic', 'lj': 'Loaminovic', 'lv': 'Luvic', 'mg': 'Magnesic', 'mf': 'Manganiferric', 'ma': 'Mahic', 'mw': 'Mawic', 'mz': 'Mazic', 'mi': 'Mineralic',
        'ml': 'Minerolimnic', 'mc': 'Mochipic', 'mo': 'Mollic', 'mm': 'Mulmic', 'mh': 'Murshic', 'mu': 'Muusic', 'nr': 'Naramic', 'na': 'Natric', 'ne': 'Nechic',
        'nb': 'Neobrunic', 'nc': 'Neocambic', 'ni': 'Nitic', 'nv': 'Novic', 'ng': 'Nudiargic', 'nt': 'Nudilithic', 'nn': 'Nudinatric', 'np': 'Nudipetric',
        'ny': 'Nudiyermic', 'oh': 'Ochric', 'ol': 'Oligoeutric', 'om': 'Ombric', 'oo': 'Organolimnic', 'ot': 'Organotransportic', 'oc': 'Ornithic', 'od': 'Orthodystric',
        'oe': 'Orthoeutric', 'of': 'Orthofluvic', 'oi': 'Orthomineralic', 'ok': 'Orthoskeletic', 'os': 'Ortsteinic', 'oa': 'Oxyaquic', 'oy': 'Oxygleyic', 'ph': 'Pachic', 
        'pb': 'Panpaic', 'vy': 'Paviyermic', 'pe': 'Pellic', 'pq': 'Pelocrustic', 'pt': 'Petric', 'pc': 'Petrocalcic', 'pd': 'Petroduric', 'pg': 'Petrogypsic', 
        'pp': 'Petroplinthic', 'ps': 'Petrosalic', 'px': 'Pisoplinthic', 'pi': 'Placic', 'pa': 'Plaggic', 'pl': 'Plinthic', 'po': 'Posic', 'pk': 'Pretic', 
        'pn': 'Profondic', 'dh': 'Profundihumic', 'pr': 'Protic', 'qa': 'Protoandic', 'qg': 'Protoargic', 'qc': 'Protocalcic', 'qy': 'Protogleyic', 
        'qq': 'Protogypsic', 'qk': 'Protokalaic', 'qz': 'Protosalic', 'qs': 'Protosodic', 'qp': 'Protospodic', 'qw': 'Protostagnic', 'qt': 'Prototechnic', 
        'qf': 'Prototephric', 'qv': 'Protovertic', 'pu': 'Puffic', 'py': 'Pyric', 'rx': 'Radiotoxic', 'rp': 'Raptic', 'ra': 'Reductaquic', 'rd': 'Reductic', 
        'rf': 'Reflectic', 'rc': 'Regic', 'rg': 'Rendzic', 're': 'Renic', 'rh': 'Rhetic', 'rm': 'Rhodic', 'rn': 'Rubic', 'ru': 'Rupic', 'sc': 'Salic', 
        'sy': 'Salinodic', 'sg': 'Saprohist', 'sl': 'Saprolithic', 'sb': 'Sapprostic', 'sm': 'Sapropelic', 'se': 'Scalpic', 'sj': 'Sculptic', 'sa': 'Selenic', 
        'sk': 'Sideralic', 'so': 'Siltic', 'sn': 'Sodic', 'sf': 'Sodifragic', 'sp': 'Spodic', 'sr': 'Stagnic', 'st': 'Stalic', 'ss': 'Stegnic', 'sx': 'Steppic', 
        'sw': 'Sulfidic', 'sh': 'Sulfiric', 'su': 'Sulfuric', 'tc': 'Technic', 'tb': 'Technicenic', 'tt': 'Tephric', 'tf': 'Terralitic', 'tl': 'Terraquantanovic', 'ti' : 'Thionic',
        'tk': 'Terric', 'tg': 'Tisovic', 'tr': 'Transportic', 'tn': 'Trunic', 'tw': 'Tuffic', 'uc': 'Ubric', 'ub': 'Umbritic', 'ue': 'Unerubic', 'uf': 'Uplimnic', 
        'um': 'Umbric', 'ur': 'Urbic', 'vl': 'Vermic', 've': 'Vertic', 'vh': 'Vitric', 'vr': 'Vulcanic', 'wh': 'Waterhostic', 'yg': 'Yermic'
    }

    # Create a new column that combines the full names for both major and minor parts
    soil_data['full_name'] = (
        soil_data['WRB_major'].map(wrb_major_names).fillna("UNKNOWN") + " " +
        soil_data['WRB_minor'].map(wrb_minor_names).fillna(soil_data['WRB_minor'])  # Keep minor as-is if not found
    )

    # Ensure 'soilType' column exists; create if not
    if 'soilType' not in df_grid.columns:
        df_grid['soilType'] = "UNKNOWN"  # Initialize with "UNKNOWN"

    # Convert data into GeoDataFrame based on coordinate positions
    data_gdf = gpd.GeoDataFrame(
        df_grid,
        geometry=gpd.points_from_xy(df_grid['longitude'], df_grid['latitude']),
        crs="EPSG:4326"  # Coordinate reference system: WGS 84
    )

    # Reproject the data to match the soil data projection, if necessary
    soil_data = soil_data.to_crs(data_gdf.crs)

    # Spatial join to find matching soil types based on location
    joined_data = gpd.sjoin(data_gdf, soil_data[['geometry', 'full_name']], how='left', predicate='intersects')

    # Assign soil types to 'soilType' column based on the join results
    df_grid['soilType'] = joined_data['full_name'].fillna("UNKNOWN")

    # Patch job because our .shp file has a few holes in it, but atleast we know what goes there
    # (Retrieved from: Soil Atlas of Europe. See page 48 for Denmark and see the first page for the soil guide)
    df_grid['soilType'] = df_grid['soilType'].replace("UNKNOWN Haplic", "Cambisol Dystric")
    df_grid['soilType'] = df_grid['soilType'].replace("UNKNOWN", "Cambisol Dystric")

    print("...soil data loaded")

    # Create positive & negative occurrence columns
    if 'positiveOccurrence' not in df_grid.columns:
        df_grid['positiveOccurrence'] = 0  # Start at 0
    if 'negativeOccurrence' not in df_grid.columns:
        df_grid['negativeOccurrence'] = 0  # Start at 0

    ### 4. Save the new template file
    df_grid.to_csv(template_name, index=False)

    print("...finished creating new template.")

# Create a new file that is an exact copy of the template
ml_file_name = 'datagrid.csv'
path = shutil.copyfile(current_dir + "\\" + template_name, current_dir + "\\" + ml_file_name)

### 5. Fill up the positive & negative occurence columns based on the original data
# Get the new data frame
df_grid = pd.read_csv(ml_file_name)

def find_nearest_grid_square(lat, lon, grid_df):
    # Find the grid square with the smallest Euclidean distance to the observation point
    distance = np.sqrt((grid_df['latitude'] - lat)**2 + (grid_df['longitude'] - lon)**2)
    nearest_idx = distance.idxmin()
    return nearest_idx

# Loop through each observation entry (from occurrences.csv), save to (ml_file_name.csv)
for _, obs in df.iterrows():
    # NOTE: Here we are searching through two different .csv files, so it's important to use the right variables
    # KEY:
    # > occurences.csv | the full data | uses: 'decimalLatitude' & 'decimalLongitude' | data frame variable: 'df'
    # > ml_file_name.csv | the simplified data field used for ML training | uses: 'latitude' & 'longitude' | data frame variable: 'df_grid'

    lat = obs['decimalLatitude']
    lon = obs['decimalLongitude']
    occurrence_status = obs['occurrenceStatus']

    # Find the nearest grid square
    nearest_idx = find_nearest_grid_square(lat, lon, df_grid)

    # Increment 'positiveOccurrence' or 'negativeOccurrence' based on 'occurrenceStatus'
    if occurrence_status == 'PRESENT':
        df_grid.at[nearest_idx, 'positiveOccurrence'] += 1
    else:
        df_grid.at[nearest_idx, 'negativeOccurrence'] += 1

print(f"(5) {ml_file_name} has been updated with position and negative occurrence data.")

### 6. Add an (empty) 'label' column
if 'label' not in df_grid.columns:
        df_grid['label'] = ""

# !!! Save the new occurrence values !!!
df_grid.to_csv(ml_file_name, index=False)
print(f"...{ml_file_name} has been saved.")

print(f"(6) Added empty 'label' column to {ml_file_name}.")

#####

print("FINISHED")