"""Convert train, test, validation data to LigthFM datasets"""
from typing import Optional, Tuple, TypeVar
from lightfm.data import Dataset
import pandas as pd

from user_recommendation import logger

COOMatrix = TypeVar("COOMatrix")


def get_lightfm_dataset(data: pd.DataFrame) -> Dataset:
    """Convert a pandas dataframe to lightfm Dataset

    Args:
        data: all interactions between users and questions

    Returns:
        Dataset: lightfm dataset
    """
    dataset = Dataset()
    logger.info("Building lightfm dataset")
    dataset.fit(
        (question_id for question_id in data.question_id.tolist()), (user_id for user_id in data.user_id.tolist())
    )
    num_question, num_user = dataset.interactions_shape()
    logger.info(f"Num question: {num_question}, num_user {num_user}.")
    return dataset


def get_interactions(
    data: pd.DataFrame, dataset: Dataset, with_weights: bool, obj_desc: Optional[str] = ""
) -> Tuple[COOMatrix, COOMatrix]:
    """Get interactions from lightfm dataset and dataframe

    Args:
        data: interactions to build
        dataset: light fm dataset
        with_weights: using weights in interactions or not

    Returns:
        Tuple[COOMatrix, COOMatrix]: interactions and weights matrix
    """
    logger.info(f"Building lightfm {obj_desc} interactions")
    return dataset.build_interactions(  # type: ignore
        (
            (row["question_id"], row["user_id"], row["score"]) if with_weights else (row["question_id"], row["user_id"])
            for index, row in data.iterrows()
        )
    )
