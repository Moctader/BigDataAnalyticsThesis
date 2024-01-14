# %%
# Import Section

import geopandas as gpd
import numpy as np
from PIL import Image
import os

# %%
# Path to the folder with all data
PREFIX = "/Users/moctader/Thesis_code"

# Zoom level
ZOOM_LEVEL = 10

# Path to the output file
DATAFRAME_OUTPUT_PATH = f"{PREFIX}/samples.pkl"

# %%
csv_path = f"{PREFIX}/GTK_ASsoil_obs.csv"
base_directory = f"{PREFIX}/output20/"

# %%

def read_geo_data(csv_path):
    points = gpd.read_file(csv_path)
    return points
points=read_geo_data(csv_path)

# %%
# Data points

points.POINT_X = points.POINT_X.astype("float")
points.POINT_Y = points.POINT_Y.astype("float")

# %%
#samples

samples = gpd.GeoDataFrame(
    points.CLASS, crs="EPSG:3067", geometry=gpd.points_from_xy(points.POINT_X, points.POINT_Y)
).to_crs("WGS84")

tile_list = [(point.x, point.y) for point in samples['geometry']]


# %%
# Creating image filename

samples["i"] = samples.index
samples["filenames"] = samples.apply(lambda row: f"{row['CLASS']}/image_{row['i']}", axis=1)

# %%
# Extracting Latitude and Longitude from GeoDataFrame


def get_lat_from_row(p):
    lon, lat = p.geometry.x, p.geometry.y
    return lat

def get_lon_from_one_column(geometry):
    lon, lat = geometry.x, geometry.y
    return lon

# %%
# Calculating Latitude for Each Sample

samples["lat"] = samples.apply(lambda row: get_lat_from_row(row), axis=1)

# %%
# Calculating Longitude for Each Sample

samples["lon"] = samples["geometry"].map(get_lon_from_one_column)


# %%
# Image Loading Function

def load_data(filename, directory):
    path = directory + "/" + filename + ".png"
    image_array = np.array(Image.open(path))

    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        return image_array[:, :, :3]
    else:
        return image_array[:, :, 0:1]

# %%
# Loading Image Data for Multiple Files

file_names = os.listdir(base_directory)

# Filter out non-directory file_names
files = [file for file in file_names if os.path.isdir(os.path.join(base_directory, file))]

for single_file in files:
    samples[single_file] = samples["filenames"].map(
        lambda name, directory=os.path.join(base_directory, single_file): load_data(name, directory)
    )

# %%
# Creating a Binary Label Column Based on "CLASS"

def label_rows(row):
    if 'ASS' == row['CLASS']:
        return 1
    else:
        return 0

samples['label'] = samples.apply(label_rows, axis=1)

# %%

# Combining Channels to Create Multi-channel Images

combined_images_list = []

# Iterate over rows of the DataFrame
for index, row in samples.iterrows():
    # Initialize an empty list to store individual layers
    layers_list = []

    # Iterate over the selected columns
    for single_file in files:
        layer = (row[single_file])
        layers_list.append(layer)

    # Concatenate the layers along the third dimension to create a multi-channel image
    combined_image = np.concatenate(layers_list, axis=-1)
    
    # Append the combined image to the list
    combined_images_list.append(combined_image)

# Add the combined images as a new column in the DataFrame

samples["combined_channels"] = combined_images_list

# %%
# Saving DataFrame as a Pickle File

samples.to_pickle(DATAFRAME_OUTPUT_PATH)


