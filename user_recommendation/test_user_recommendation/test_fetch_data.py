"""Test data fetching"""
from pathlib import Path
from unittest import TestCase

import pandas as pd

from user_recommendation.data_preparation.data_loading.fetch_data import read_data
from user_recommendation.errors import InputDataError


class ReadDataTest(TestCase):
    """Unittest on utils function"""

    def test_read_data(self) -> None:
        """Test function read_data"""

        df = read_data("test.json", Path(__file__).parent / "data")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (3, 6))

    def test_read_data_fail(self) -> None:
        """Test function read_data"""

        with self.assertRaises(InputDataError):
            read_data("test_a.json", Path(__file__).parent / "data")
