import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from create_Dataframe_unit_test import (
    read_geo_data,

)

class TestYourModule(unittest.TestCase):
    def setUp(self):
        # Mock GeoDataFrame for testing
        self.mock_geo_data = pd.DataFrame({
            'POINT_X': [1.0, 2.0],
            'POINT_Y': [3.0, 4.0],
            'CLASS': ['A', 'B'],
        })

        # Mock samples GeoDataFrame for testing
        self.mock_samples = pd.DataFrame({
            'CLASS': ['A', 'B'],
            'geometry': [Mock(), Mock()],
        })

        # Mock base directory for testing
        self.mock_base_directory = '/mock/base_directory/'

        # Mock images directory for testing
        self.mock_images_directory = '/mock/images_directory/'

        # Mock save path for testing
        self.mock_save_path = '/mock/save_path.pkl'

    def test_read_geo_data(self):
        with patch('geopandas.read_file', return_value=self.mock_geo_data):
            result = read_geo_data('/mock/csv_path/')
        self.assertTrue(result.equals(self.mock_geo_data))