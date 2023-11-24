import numpy as np
import geopandas as gpd
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import math

output_root_directory = "/Users/moctader/Thesis_code/output4/"
points = gpd.read_file("/Users/moctader/Thesis_code/GTK_ASsoil_obs.csv")
points.POINT_X = points.POINT_X.astype("float")
points.POINT_Y = points.POINT_Y.astype("float")

samples = gpd.GeoDataFrame(
    points.CLASS, crs="EPSG:3067", geometry=gpd.points_from_xy(points.POINT_X, points.POINT_Y)
).to_crs("WGS84")


tile_list = [(point.x, point.y) for point in samples['geometry']]

def calculate_distance(tile1, tile2):
    x1, y1 = tile1
    x2, y2 = tile2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def find_nearest_tile(tile, tile_list):
    distances = {}

    for other_tile in tile_list:
        if other_tile != tile:
            distance = calculate_distance(tile, other_tile)
            distances[other_tile] = distance

    # Find the nearest tile
    nearest_tile = min(distances, key=distances.get)
    return nearest_tile, distances[nearest_tile]


TILE_SIZE = 256


def project(p, zoom):
    lon, lat = p.geometry.x, p.geometry.y

    siny = np.sin(lat * np.pi / 180)
    siny = min(max(siny, -0.9999), 0.9999)

    x = TILE_SIZE * (0.5 + lon / 360)
    y = TILE_SIZE * (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi))

    scale = 2**zoom

    tx = x * scale // TILE_SIZE
    ty = y * scale // TILE_SIZE

    px = x * scale % TILE_SIZE // 1
    py = y * scale % TILE_SIZE // 1

    return (int(zoom), int(tx), int(ty)), (px, py)


nearest_tile_dict = {}

for tile in tile_list:
    nearest_tile, distance = find_nearest_tile(tile, tile_list)
    nearest_tile_dict[tile] = {'nearest_tile': nearest_tile, 'distance': distance}


def calculate_neighboring_tiles(tx, ty, radius=1):
    neighboring_tiles = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            neighboring_tiles.append((tx + dx, ty + dy))
    return neighboring_tiles




def combine_tiles(p, zoom, tile_template, t_value, nearest_tile_dict):
    (tz, tx, ty), (px, py) = project(p, zoom=zoom)

    # Get a list of all neighboring tiles
    radius = 1 
    neighboring_tiles = calculate_neighboring_tiles(tx, ty, radius)

   # Adjust the range as needed
    extra_left_tiles = [(tx-i, ty) for i in range(4, 8)]  
    neighboring_tiles.extend(extra_left_tiles)


    # Load neighboring tiles and find the closest tile
    tile_images = {}
    closest_tile = None
    closest_distance = float('inf')

    for tile in neighboring_tiles:
        nx, ny = tile
        fname = tile_template.format(t=t_value, z=tz, x=nx, y=ny)
        try:
            tile_image = Image.open(fname)
            tile_images[tile] = tile_image
        except FileNotFoundError:
            pass

    # Check if there are tiles available before proceeding
    if not tile_images:
        print("No tiles available.")
        return None, None, None, None

    # Assuming each tile has the same dimensions
    width, height = list(tile_images.values())[0].size

    # Create a new image with dimensions based on the number of tiles
    combined_image = Image.new('RGB', (3 * width, 3 * height))

    # Paste each tile into the combined image at its respective position
    # Adjust position based on central tile
    for tile, tile_image in tile_images.items():
        nx, ny = tile
        x, y = nx - tx + 1, ny - ty + 1  
        combined_image.paste(tile_image, (x * width, y * height))

 
    center_x, center_y = px, py
    crop_size = 50

    # Calculate the crop box
    left = max(0, center_x - crop_size // 2)
    top = max(0, center_y - crop_size // 2)
    right = min(combined_image.width, left + crop_size)
    bottom = min(combined_image.height, top + crop_size)

    # Crop the region of interest
    cropped_image = combined_image.crop((left, top, right, bottom))

    # plt.figure(figsize=(18, 5))  # Adjust the figure size as needed

    # # Original Combined Image
    # plt.subplot(1, 2, 1)

    # plt.imshow(combined_image, interpolation="lanczos")
    # plt.plot(px, py, "*", c="r", markersize=12)
    # plt.text(px, py, f'({px}, {py})', color="r", fontsize=8, verticalalignment='bottom',
    #          horizontalalignment='right')
    # plt.title("Original Combined Image")
    
    
    # plt.subplot(1, 2,2)
    # # 50x50 Image Centered around px, py
    # plt.imshow(cropped_image, interpolation="lanczos")
    # plt.plot(center_x - left, center_y - top, "*", c="r", markersize=12)
    # plt.text(center_x - left, center_y - top, f'({center_x}, {center_y})', color="r", fontsize=8,
    #          verticalalignment='bottom', horizontalalignment='right')
    # plt.title("50x50 Image Centered around px, py")

    # plt.show()

    # Save the cropped image in a subdirectory based on the variable 't'
    output_directory = os.path.join(output_root_directory, t_value)
    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory, f"image_{p.name}.png")
    cropped_image.save(output_path)

    return cropped_image, tz, tx, ty
