import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
import os


PREFIX = "Users/moctader/Thesis_code"  # folder with files
ZOOM_LEVEL = 10  # Zoom level
DATAFRAME_OUTPUT_PATH = f"{PREFIX}/samples.pkl"  # Path to the output file

csv_path = f"{PREFIX}/GTK_ASsoil_obs.csv"
base_directory = f'{PREFIX}/output20/'


def read_geo_data(csv_path: str) -> gpd.GeoDataFrame:
    points = gpd.read_file(csv_path)
    return points

def convert_and_create_geodataframe(points: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Convert and create a GeoDataFrame from a DataFrame of points.
    
    Args:
        points (pandas.DataFrame): DataFrame containing point data.
    
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame with converted columns and geometry.
    """

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

def add_filenames(samples: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    samples["i"] = samples.index
    samples["filenames"] = samples.apply(lambda row: f"{row['CLASS']}/image_{row['i']}", axis=1)
    return samples

def add_lat_lon(samples: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    samples["lat"] = samples.apply(lambda row: row.geometry.y, axis=1)
    samples["lon"] = samples.apply(lambda row: row.geometry.x, axis=1)
    return samples

def load_data(filename: str, directory: str) -> np.ndarray:
    path = os.path.join(directory, filename + ".png")
    image_array = np.array(Image.open(path))
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        return image_array[:, :, :3]
    else:
        return image_array[:, :, 0:1]

def add_images(samples: gpd.GeoDataFrame, base_directory: str) -> gpd.GeoDataFrame:
    """
    Add images to the samples DataFrame.

    Args:
        samples (DataFrame): The DataFrame to which the images will be added.
        base_directory (str): The base directory containing the image files.

    Returns:
        DataFrame: The updated samples DataFrame with images added.
    """
    file_names = os.listdir(base_directory)
    files = [file for file in file_names if os.path.isdir(os.path.join(base_directory, file))]

    for single_file in files:
        samples[single_file] = samples["filenames"].map(
            lambda name, directory=os.path.join(base_directory, single_file): load_data(name, directory)
        )

    return samples

def add_labels(samples: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    samples["label"] = samples.apply(lambda row: 1 if row['CLASS'] == 'ASS' else 0, axis=1)
    return samples

def combine_channels(samples: gpd.GeoDataFrame, base_directory: str) -> gpd.GeoDataFrame:
    """
    Combine individual channels of images into a multi-channel image.

    Args:
        samples (gpd.GeoDataFrame): The input GeoDataFrame containing the image data.
        base_directory (str): The base directory where the individual channel files are stored.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with a new column "combined_channels" containing the combined images.
    """    
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


def save_dataframe(samples: gpd.GeoDataFrame, save_path: str):
    samples.to_pickle(save_path)


def main():
    points = read_geo_data(csv_path)
    points_wgs84 = convert_and_create_geodataframe(points)
    samples = add_filenames(points_wgs84)
    samples = add_lat_lon(samples)
    samples = add_images(samples, base_directory)
    samples = add_labels(samples)
    samples = combine_channels(samples, base_directory)
    save_dataframe(samples, DATAFRAME_OUTPUT_PATH)


if __name__ == "__main__":
    main()
