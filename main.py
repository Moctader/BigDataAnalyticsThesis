import numpy as np
import geopandas as gpd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from preprocessing import samples, combine_tiles, nearest_tile_dict
import glob
import os


tiles_template = "/Users/moctader/Thesis/{t}/{z}/{x}/{y}.png"
zoom_level = 10
t_values = glob.glob('/Users/moctader/Thesis/*')

# unique_tiles = set(tile for tile, info in nearest_tile_dict.items())
# sorted_tiles = sorted(nearest_tile_dict.items(), key=lambda x: x[1]['distance'])

for i in range(samples.shape[0]):
    for t_value in t_values:
        if not t_value.endswith('.zip'):
            t_value = os.path.splitext(os.path.basename(t_value))[0]

            try:
                cropped_image, tz, tx, ty = combine_tiles(samples.iloc[i], zoom_level, tiles_template, t_value,
                                                                 samples['CLASS'][i])

            except FileNotFoundError:
                print(f"No valid {t_value} found for sample {i}  ------> {tz}/{tx}/{ty}  ")
                pass