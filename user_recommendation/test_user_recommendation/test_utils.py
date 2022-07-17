"""Test utils functions"""
from enum import Enum
import math
from unittest import TestCase
import unittest

import pandas as pd
import numpy as np

from user_recommendation import logger
from user_recommendation.utils import string_to_enum
from user_recommendation.errors import InvalidTag


class EnumTest(Enum):
    """Test Enum

    Args:\
        Enum (str): test enum
    """

    SPLIT_DATASET = "split_dataset"


class UtilsFunctionsTest(TestCase):
    """Unittest on utils function"""

    def test_string_to_enum(self) -> None:
        """Test function string_to_enum"""

        string_enum_valid = "split_dataset"

        err_cls = InvalidTag

        enum_valid = string_to_enum(enum=string_enum_valid, enum_class=EnumTest, err_cls=err_cls, logger=logger)
        self.assertIsInstance(enum_valid, EnumTest)
        self.assertEqual(enum_valid.name, EnumTest(string_enum_valid).name)

    def test_string_to_enum_fail(self) -> None:
        """Test function string_to_enum"""

        string_enum_invalid = "cnn"

        err_cls = InvalidTag

        with self.assertRaises(InvalidTag):
            string_to_enum(enum=string_enum_invalid, enum_class=EnumTest, err_cls=err_cls, logger=logger)

    # def test_log_channels_per_cat(self) -> None:
    #     """Test function log channels per cat"""

    #     ids_videos = ["id_video1", "id_video2", "id_video3", "id_video4", "id_video5", "id_video6", "id_video7"]
    #     ids_channels = [
    #         "id_channel1",
    #         "id_channel1",
    #         "id_channel2",
    #         "id_channel2",
    #         "id_channel2",
    #         "id_channel3",
    #         "id_channel_3",
    #     ]
    #     category_mere = ["sport", "sport", "sport", "sport", "sport", "food", "food"]
    #     yt_category_label = ["Sport", "Sport", "Sport", "Sport", "Sport", "Foods", "Foods"]
    #     channel_label = [
    #         "football",
    #         "football",
    #         "us football",
    #         "us football",
    #         "us football",
    #         "street_food",
    #         "street_food",
    #     ]

    #     df_data = pd.DataFrame(
    #         {
    #             "video_id": ids_videos,
    #             "channel_id": ids_channels,
    #             "yt_category_label": yt_category_label,
    #             "channel_label": channel_label,
    #             "category_mere": category_mere,
    #         }
    #     )
    #     attribute = "channel_label"
    #     desc = "unittest on log channels per cat"

    #     log_channels_per_cat(df_data=df_data, attribute=attribute, desc=desc, logger=logger)
