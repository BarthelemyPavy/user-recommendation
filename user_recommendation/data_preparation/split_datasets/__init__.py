"""Module for train test val datasets generation"""
from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Union, overload
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split


class Split(ABC):
    """Abstract Class for train test val split

    Attributes:
        _val_size: Rate of validation from test_val dataset
    """

    _val_size: float = 0.5

    def __init__(self, preprocessed_df: pd.DataFrame) -> None:
        """Class constructor

        Args:
            preprocessed_df: DataFrame preprocessed by previous step, see PreprocessDataframes for more information
        """
        self._preprocessed_df = preprocessed_df

    def _deduplicate_questions(self) -> pd.DataFrame:
        """From preprocessed df keep only unique question ids

        Returns:
            pd.DataFrame: dataframe with question_id as primary key
        """
        return self._preprocessed_df.drop_duplicates(subset="question_id").reset_index(drop=True)

    @classmethod
    def _test_val_split(
        cls, unique_questions: pd.DataFrame, first_split_size: Union[float, int], random_state: int
    ) -> Tuple[list[int], list[int]]:
        """Split questions into test and val ids

        Args:
            unique_questions: DataFrame with question_id as primary key
            first_split_size: Size of first split
            random_state: random_state for split reproducibility

        Returns:
            Tuple[list[str], list[str]]: tuple format as (test_question_ids, val_question_ids)
        """
        # Split questions in test_eval dataset
        _, test_val_questions = train_test_split(
            unique_questions[["question_id", "question_label"]],
            test_size=first_split_size,
            random_state=random_state,
            stratify=unique_questions["question_label"],
        )

        # Split test_eval dataset into test and eval
        test, val = train_test_split(
            test_val_questions[["question_id"]],
            test_size=cls._val_size,
            random_state=random_state,
            stratify=test_val_questions["question_label"],
        )
        return (test, val)

    # For **kwargs typing https://peps.python.org/pep-0484/#arbitrary-argument-lists-and-default-argument-values
    @abstractmethod
    def execute(self, **kwargs: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main function of the class

        Args:
            split_size: Rate of split for test_val datasets
            random_state: random_state for split reproducibility

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: tuple of dataframes containing test dataframe and val dataframe,
                                                (test_df, val_df)
        """


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
