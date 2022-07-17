"""Object packing train test and validation datasets"""
from typing import Iterator
from dataclasses import dataclass
import pandas as pd


@dataclass
class Datasets:
    """Split dataset for supervised Machine Learning into training, validation and test

    Attributes:\
        training: training dataset
        test: test dataset
        validation: validation dataset
    """

    training: pd.DataFrame
    test: pd.DataFrame
    validation: pd.DataFrame

    def __iter__(self) -> Iterator[tuple[str, pd.DataFrame]]:
        """Iterate over Dataset object

        Yield:\
            Iterator[tuple[str, pd.DataFrame]]: Dataframe contained in Dataset object
        """
        for attr, value in self.__dict__.items():
            yield attr, value

    def asdict(self) -> dict[str, pd.DataFrame]:
        """Convert to dictionary

        Returns:\
            dict[str, pd.DataFrame]: Dataset attribute converted to dict
        """
        return vars(self)


@dataclass
class DataArtifacts:
    """Paths to artifacts use for prediciton

    Attributes:\
        path_to_datasets: path to train, test, validation datasets
    """

    path_to_datasets: str

    def asdict(self) -> dict[str, str]:
        """Convert to dictionary

        Returns:\
            dict[str, str]: DataArtifacts attribute as dictionary
        """
        return vars(self)


SERIALIZED_DATA_ARTIFACT_NAMES = DataArtifacts(path_to_datasets="data/datasets/")
