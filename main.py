import numpy as np
import geopandas as gpd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from preprocessing import display_pixel, samples



tiles = "/Users/moctader/Thesis/{t}/{z}/{x}/{y}.png"

zoom_level = 10
t_values = ['landsat', 'aem_apparent_resistivity', 'aem_imaginary_component',
            'aem_real_component', 'elev_10m', 'elev_10m_aspect', 'elev_10m_hillshade',
            'elev_10m_slope', 'TPI', 'corine_land_cover']

for i in range(samples.shape[0]):
    for t_value in t_values:
        try:
            display_pixel(samples.iloc[i], zoom_level, tiles, t_value)
        except FileNotFoundError:
            print(f"No valid {t_value} found for sample {i}")
            pass
     