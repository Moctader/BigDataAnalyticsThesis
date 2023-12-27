import argparse
import geopandas as gpd
import pandas as pd
import numpy as np
from PIL import Image
import os


csv_path = '/Users/moctader/Thesis_code/trunk_based_thesis/another/GTK_ASsoil_obs.csv'
base_directory = '/Users/moctader/Thesis_code/output20/'
save_path = '/Users/moctader/Thesis_code/trunk_based_thesis/another/samples.pkl'


def read_geo_data(csv_path):
    points = gpd.read_file(csv_path)
    return points

def convert_and_create_geodataframe(points):
    # Convert columns to float
    points['POINT_X'] = points['POINT_X'].astype("float")
    points['POINT_Y'] = points['POINT_Y'].astype("float")

    # Create GeoDataFrame with specified CRS and geometry
    samples = gpd.GeoDataFrame(
        points['CLASS'],
        crs="EPSG:3067",
        geometry=gpd.points_from_xy(points['POINT_X'], points['POINT_Y'])
    )

    # Transform to WGS84
    samples = samples.to_crs("EPSG:4326")

    return samples

def add_filenames(samples):
    samples["i"] = samples.index
    samples["filenames"] = samples.apply(lambda row: f"{row['CLASS']}/image_{row['i']}", axis=1)
    return samples

def add_lat_lon(samples):
    samples["lat"] = samples.apply(lambda row: row.geometry.y, axis=1)
    samples["lon"] = samples.apply(lambda row: row.geometry.x, axis=1)
    return samples

def load_data(filename, directory):
    path = os.path.join(directory, filename + ".png")
    image_array = np.array(Image.open(path))
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        return image_array[:, :, :3]
    else:
        return image_array[:, :, 0:1]

def add_images(samples, base_directory):
    file_names = os.listdir(base_directory)
    files = [file for file in file_names if os.path.isdir(os.path.join(base_directory, file))]

    for single_file in files:
        samples[single_file] = samples["filenames"].map(
            lambda name, directory=os.path.join(base_directory, single_file): load_data(name, directory)
        )

    return samples

def add_labels(samples):
    samples["label"] = samples.apply(lambda row: 1 if row['CLASS'] == 'ASS' else 0, axis=1)
    return samples

def combine_channels(samples, base_directory):
    # Get the list of columns that represent individual channels
    files = [file for file in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, file))]

    # Initialize a list to store the combined images
    combined_images_list = []

    # Iterate over rows of the DataFrame
    for index, row in samples.iterrows():
        # Initialize an empty list to store individual layers
        layers_list = []

        # Iterate over the selected columns
        for single_file in files:
            layer = row[single_file]
            layers_list.append(layer)

        # Concatenate the layers along the third dimension to create a multi-channel image
        combined_image = np.concatenate(layers_list, axis=-1)

        # Append the combined image to the list
        combined_images_list.append(combined_image)

    # Add the combined images as a new column in the DataFrame
    samples["combined_channels"] = combined_images_list

    return samples


def save_dataframe(samples, save_path):
    samples.to_pickle(save_path)

def main():


    points = read_geo_data(csv_path)
    points_wgs84 = convert_and_create_geodataframe(points)
    samples = add_filenames(points_wgs84)
    samples = add_lat_lon(samples)
    samples = add_images(samples, base_directory)
    samples = add_labels(samples)
    samples = combine_channels(samples, base_directory)
    save_dataframe(samples, save_path)

if __name__ == "__main__":
    main()
