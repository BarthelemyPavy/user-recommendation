"""Fetch data process, from google drive download to load as dataframe"""
import os
from pathlib import Path
from typing import Union
import zipfile
import gdown
import pandas as pd

from user_recommendation import logger
from user_recommendation.utils import log_raise
from user_recommendation.errors import InputDataError

AVAILABLE_FILES = ["answers.json", "questions.json", "users.json"]


def download_data(
    path: Union[str, Path] = Path(__file__).parent / ".." / ".." / ".." / ".." / "data" / "data.zip"
) -> None:
    """Download and unzip project data

    Args:
        path: Path to store data
    """

    url = "https://drive.google.com/u/0/uc?id=1CUcfl3JX8TNYABn2JRIPQozT0oqdqqOy&export=download"
    logger.info(f"Download data from {url}")
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        gdown.download(url=url, output=str(path), quiet=False, fuzzy=True)

        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(Path(path).parent)
        os.remove(path)
    except Exception as err:
        log_raise(logger=logger, err=InputDataError(), original_err=err)
    logger.info("Data succesfully downloade")


def read_data(
    file_name: str, path: Union[str, Path] = Path(__file__).parent / ".." / ".." / ".." / ".." / "data"
) -> pd.DataFrame:
    """Read json file

    Args:
        file_name: Name of the file to read

    Returns:
        pd.DataFrame: DataFrame containing data from json
    """
    try:
        df = pd.read_json(str(path / Path(file_name)), lines=True)
    except (ValueError, KeyError, pd.core.base.DataError) as err:
        log_raise(logger=logger, err=InputDataError(), original_err=err)
    logger.info(f"{file_name} loaded, shape of dataframe: {df.shape}")
    return df
