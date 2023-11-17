import numpy as np
import geopandas as gpd
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


output_root_directory = "/Users/moctader/Thesis_code/output3/"
points = gpd.read_file("/Users/moctader/Thesis_code/GTK_ASsoil_obs.csv")
points.POINT_X = points.POINT_X.astype("float")
points.POINT_Y = points.POINT_Y.astype("float")

samples = gpd.GeoDataFrame(
    points.CLASS, crs="EPSG:3067", geometry=gpd.points_from_xy(points.POINT_X, points.POINT_Y)
).to_crs("WGS84")

#print(samples.head())




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




def display_pixel(p, zoom, tileset, t_value, window_size=50):
    (tz, tx, ty), (px, py) = project(p, zoom=zoom)

    fname = tileset.format(t=t_value, z=tz, x=tx, y=ty)
    im = Image.open(fname)

    # Ensure the cropped window is centered around the point
    left = max(px - window_size // 2, 0)
    top = max(py - window_size // 2, 0)
    right = min(px + window_size // 2, TILE_SIZE)
    bottom = min(py + window_size // 2, TILE_SIZE)

    # Calculate the adjustment to center the point
    adjustment_x = (window_size - (right - left)) // 2
    adjustment_y = (window_size - (bottom - top)) // 2

    # Apply the adjustment
    left -= adjustment_x
    right += adjustment_x
    top -= adjustment_y
    bottom += adjustment_y

    # Crop the image
    im_cropped = im.crop((left, top, right, bottom))

    # Save the cropped image in a subdirectory based on the variable 't'
    output_directory = os.path.join(output_root_directory, t_value)
    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory, f"image_{p.name}.png")
    im_cropped.save(output_path)

    # Display the cropped image with the point at the center
    plt.imshow(im_cropped, interpolation="lanczos")
    plt.plot(window_size // 2, window_size // 2, "*", c="c", markersize=12)  
    #plt.show()
