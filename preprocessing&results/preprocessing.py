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
import csv
from PIL import ImageDraw

# Read the files
PREFIX = "/Users/moctader/Arcada/"
output_path=f"{PREFIX}/check_data"
output_root_directory = f"{PREFIX}/check_data"
points = gpd.read_file(f"{PREFIX}/GTK_ASsoil_obs.csv")
points.POINT_X = points.POINT_X.astype("float")
points.POINT_Y = points.POINT_Y.astype("float")

#samples
samples = gpd.GeoDataFrame(
    points.CLASS, crs="EPSG:3067", geometry=gpd.points_from_xy(points.POINT_X, points.POINT_Y)
).to_crs("WGS84")

# Tile size
TILE_SIZE = 256

def project(p, zoom, point_class):
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

    return (int(zoom), int(tx), int(ty)), (px, py), point_class



def calculate_neighboring_tiles(tx, ty, radius=1):
    neighboring_tiles = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            neighboring_tiles.append((tx + dx, ty + dy))
    return neighboring_tiles



def combine_tiles(p, zoom, tile_template, t_value, point_class):
    (tz, tx, ty), (px, py), _ = project(p, zoom=zoom, point_class=point_class)

    # Check if the tile is available
    actual_fname = tile_template.format(t=t_value, z=tz, x=tx, y=ty)
    if not os.path.isfile(actual_fname):
        with open('output.txt', 'a') as f:
            f.write(f"{p.name}\n")
            f.write(f"{actual_fname}\n")
        return None
        
    else:
        # Get a list of all neighboring tiles
        radius = 1 
        neighboring_tiles = calculate_neighboring_tiles(tx, ty, radius)
        

        # Load the images of the neighboring tiles
            
        tile_images = {}
        for tile in neighboring_tiles:
            nx, ny = tile
            fname = tile_template.format(t=t_value, z=tz, x=nx, y=ny)
            if(os.path.isfile(fname)):
                tile_image = Image.open(fname)
                tile_images[tile] = tile_image
            else:
                pass

        # Get the dimensions of the tiles
        width, height = list(tile_images.values())[0].size

        # Create a new image with dimensions based on the maximum tile coordinates
        max_x = max(x for x, y in tile_images.keys())
        min_x = min(x for x, y in tile_images.keys())
        max_y = max(y for x, y in tile_images.keys())
        min_y = min(y for x, y in tile_images.keys())
        combined_image = Image.new('RGB', ((max_x - min_x + 1) * width, (max_y - min_y + 1) * height))

        # Paste each tile into the combined image at its respective position
        for (x, y), tile_image in tile_images.items():
            combined_image.paste(tile_image, ((x - min_x) * width, (y - min_y) * height))

        # # Create a draw object
        # draw = ImageDraw.Draw(combined_image)


        # Adjust (px, py) based on the position of the tile (tx, ty) within the combined image
        px += (tx - min_x) * width
        py += (ty - min_y) * height

        center_x, center_y = px, py
        crop_size = 50
        
        # Calculate the crop box coordinates
        left = max(0, px - crop_size // 2)
        upper = max(0, py - crop_size // 2)
        right = min(combined_image.width, px + crop_size // 2 + crop_size % 2)
        lower = min(combined_image.height, py + crop_size // 2 + crop_size % 2)

        # Calculate the amount of pixels needed to fill on each side
        left_fill = max(0, crop_size - (right - left))
        right_fill = max(0, crop_size - (right - left))
        top_fill = max(0, crop_size - (lower - upper))
        bottom_fill = max(0, crop_size - (lower - upper))

        # Adjust the crop box coordinates to fill missing pixels
        left = max(0, left - right_fill)
        right = min(combined_image.width, right + left_fill)
        upper = max(0, upper - bottom_fill)
        lower = min(combined_image.height, lower + top_fill)

        # Crop the image
        cropped_image = combined_image.crop((left, upper, right, lower))
        center_pixel_value = cropped_image.getpixel((center_x - left, center_y - upper))


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
        # plt.plot(center_x - left, center_y - upper, "*", c="r", markersize=12)
        # plt.text(center_x - left, center_y - upper, f'({center_x}, {center_y})', color="r", fontsize=8,
        #          verticalalignment='bottom', horizontalalignment='right')
        # plt.title("50x50 Image Centered around px, py")
        # plt.show()

        # Save the cropped image in a subdirectory based on the variable 't'
        output_directory = os.path.join(output_root_directory, str(zoom), t_value, point_class)
        os.makedirs(output_directory, exist_ok=True)
        output_path = os.path.join(output_directory, f"image_{p.name}.png")
        cropped_image.save(output_path)
    
        return 