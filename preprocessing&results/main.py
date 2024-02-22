import numpy as np
import geopandas as gpd
from PIL import Image
import matplotlib.pyplot as plt
from preprocessing import samples, combine_tiles, project
import glob
import os
import csv
from PIL import ImageDraw

# Read the files
tiles_template = "/Users/moctader/all_data/{t}/{z}/{x}/{y}.png"
t_values = glob.glob('/Users/moctader/all_data/*')

# This section of the code only to generate whole map (Draw the points on the map)
def draw_points_on_image(image, points, point_classes, marker_size=2):
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size

    for (px, py), pc in zip(points, point_classes):
        if 0 <= px < img_width and 0 <= py < img_height:
            marker_color = 'blue' if pc == 'ASS' else 'red'
            marker_position = (px - marker_size // 2, py - marker_size // 2, px + marker_size // 2, py + marker_size // 2)
            draw.ellipse(marker_position, fill=marker_color)

    return image

# Main function to call the all the values
for zoom_level in range(2,16):  
    # positions = []
    # point_classes = []
    for i in range(samples.shape[0]):
        for t_value in t_values:
            if not t_value.endswith('.zip'):
                t_value = os.path.splitext(os.path.basename(t_value))[0]
                try:
                    fname=combine_tiles(samples.iloc[i], zoom_level, tiles_template, t_value, samples['CLASS'][i])
                    if fname is None:
                        continue
                    # positions.append((px, py))
                    # point_classes.append(point_class)
                except FileNotFoundError:
                    pass
                

#Use the draw function to create the image(it will only work for upto the zoom level 5)

#zoom_level = 5
#combined_image = draw_points_on_image(combine_image, positions, point_classes)
#combined_image.save('combined_image.png')