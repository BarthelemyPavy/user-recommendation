"""Test utils functions"""
from enum import Enum
from unittest import TestCase

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
