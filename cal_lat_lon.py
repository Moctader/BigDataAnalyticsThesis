# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import glob
import os
import math
import csv


# %%
lat_lon_file_path='/Users/moctader/Thesis_code/lat_lon/results.csv'
feature_vector_path='/Users/moctader/Thesis_code/lat_lon/feature_vector.csv'
csv_file_path = '/Users/moctader/Thesis_code/lat_lon/results.csv'
tiles_template = "/Users/moctader/Thesis/{t}/{z}/{x}/{y}.png"
combined_csv_path = '/Users/moctader/Thesis_code/lat_lon/combined_results.csv'
points = gpd.read_file("/Users/moctader/Thesis_code/GTK_ASsoil_obs.csv")
t_values = glob.glob('/Users/moctader/Thesis/*')


# %%
points.POINT_X = points.POINT_X.astype("float")
points.POINT_Y = points.POINT_Y.astype("float")
zoom_level = 10


# %%
samples = gpd.GeoDataFrame(
    points.CLASS, crs="EPSG:3067", geometry=gpd.points_from_xy(points.POINT_X, points.POINT_Y)
).to_crs("WGS84")

tile_list = [(point.x, point.y) for point in samples['geometry']]

# %%

results_list = []

def project(p, zoom, point_class):
    lon, lat = p.geometry.x, p.geometry.y
    results_list.append([lon, lat])

    siny = np.sin(lat * np.pi / 180)
    siny = min(max(siny, -0.9999), 0.9999)

    x = TILE_SIZE * (0.5 + lon / 360)
    y = TILE_SIZE * (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi))

    scale = 2**zoom

    tx = x * scale // TILE_SIZE
    ty = y * scale // TILE_SIZE

    px = x * scale % TILE_SIZE // 1
    py = y * scale % TILE_SIZE // 1

    return (int(zoom), int(tx), int(ty)), (px, py), point_class

# %%


TILE_SIZE = 256

for i in range(samples.shape[0]):
    for t_value in t_values:
        if not t_value.endswith('.zip'):
            t_value = os.path.splitext(os.path.basename(t_value))[0]
           
            try:
                project(samples.iloc[i], zoom_level, samples['CLASS'][i])
                
               
            except FileNotFoundError:
                print(f"No valid {t_value} found for sample {i}  ------> {tz}/{tx}/{ty}  ")
                pass
            
            

# %%
columns = ['lon', 'lat']
# Convert the results_list to a DataFrame
df = pd.DataFrame(results_list, columns=columns)

# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)
results_list
print(df)

# %%
# Load the data from CSV files into DataFrames
df1 = pd.read_csv(lat_lon_file_path)
df2 = pd.read_csv(feature_vector_path)

# Concatenate the two DataFrames along columns
combined_df = pd.concat([df2, df1], axis=1)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv(combined_csv_path, index=False)

# Display the combined DataFrame
print(combined_df.head(2))

# %%




