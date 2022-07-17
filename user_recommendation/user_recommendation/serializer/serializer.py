from enum import Enum
import json
import logging
from pathlib import Path
import pickle
from typing import Optional, TypeVar

import pandas as pd
from user_recommendation import logger
from user_recommendation.utils import log_raise

T = TypeVar("T")


class ESerializerExtension(Enum):
    """List of extension handle by serializer

    Attributes
        CSV: csv extension
        PKL: pickle extension

    """

    CSV = "csv"
    PKL = "pkl"
    JSON = "json"


class Serializer:
    """Class to serialize object to s3 with several extension.

    Attributes:\
        _path: See __call__ doc
        _logger: project logger use to log to cloudwatch
    """

    _logger: logging.Logger
    _path: str

    def __init__(self) -> None:
        """Serializer constructor"""
        self._logger = logger

    def __call__(
        self, to_serialize: T, path: str, extension: ESerializerExtension, obj_desc: Optional[str] = None
    ) -> None:
        """Serialize input object

        Args:\
            to_serialize: Object to serialize
            path: Path to serialize object
            extension: Object extension, for more information go to ESerializerExtension class
            obj_desc: Description of item to serialize for logs
        """
        self._path = path
        self._check_path_and_create()
        self._log_serialization(obj_desc, extension)
        try:
            if extension == ESerializerExtension.CSV:
                self._csv_serialization(to_serialize)
            elif extension == ESerializerExtension.JSON:
                self._json_serialization(to_serialize)
            elif extension == ESerializerExtension.PKL:
                self._pickle_serialization(to_serialize)
            else:
                raise ValueError("Bad extension type, please check ESerializerExtension class")
        except ValueError as err:
            log_raise(logger=self._logger, err=err)
        self._logger.info("Object correctly stored")

    def _check_path_and_create(self) -> None:
        """Check if directory path exists and create it if not"""
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)

    def _pickle_serialization(self, data_to_serialize: T) -> None:
        """Serialize input object with pickle extension

        Args:\
            data_to_serialize: Data to upload to s3 with pickle extension
        """
        with open(self._path, "wb") as output_file:
            pickle.dump(data_to_serialize, output_file)

    def _csv_serialization(self, data_to_serialize: T) -> None:
        """Serialize input object with csv extension

        Args:\
            data_to_serialize: Data to upload to s3 with csv extension
        """
        if not isinstance(data_to_serialize, pd.DataFrame):
            raise ValueError("CSV extension type is allowed for pandas DataFrame only")
        data_to_serialize.to_csv(self._path, index=False)

    def _json_serialization(self, data_to_serialize: T) -> None:
        """Serialize input object with json extension

        Args:\
            data_to_serialize: Data to upload to s3 with json extension
        """
        if not isinstance(data_to_serialize, dict):
            raise ValueError("JSON extension type is allowed for dictionary only")
        with open(self._path, "w") as output_file:
            json.dump(data_to_serialize, output_file)

    def _log_serialization(self, obj_desc: Optional[str], extension: ESerializerExtension) -> None:
        """Log information about serialization with helper.

        Args:\
            obj_desc: Description of item to serialize for logs
            extension: Object extension, for more information go to ESerializerExtension class
        """
        to_log = f"Serialize object to {Path(self._path).resolve()} as {extension}."
        if obj_desc:
            to_log += f" Object description: {obj_desc}"
        self._logger.info(to_log)
