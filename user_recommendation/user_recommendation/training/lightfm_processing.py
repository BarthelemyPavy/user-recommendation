"""Convert train, test, validation data to LigthFM datasets"""
from typing import Optional, Tuple, TypeVar
from lightfm.data import Dataset
import pandas as pd

from user_recommendation import logger
from user_recommendation.utils import log_raise

COOMatrix = TypeVar("COOMatrix")

USERS_NUM_COL: list[str] = ["reputation", "up_votes", "down_votes", "views"]


def get_lightfm_dataset(
    users: pd.DataFrame,
    questions: pd.DataFrame,
    user_features: Optional[pd.DataFrame] = None,
    question_features: Optional[pd.DataFrame] = None,
    tags_columns: Optional[list[str]] = None,
) -> Dataset:
    """Convert a pandas dataframe to lightfm Dataset

    Args:
        users: Dataframe containing all user ids
        questions: Dataframe containing all question ids
        user_features: Dataframe containing user features
        question_features: Dataframe containing question features
        tags_columns: Name of columns extracted from text processing.
                    Required if question_features is set passed

    Returns:
        Dataset: lightfm dataset
    """
    dataset = Dataset()
    logger.info("Building lightfm dataset")
    dataset.fit(
        (question_id for question_id in questions.question_id.tolist()), (user_id for user_id in users.id.tolist())
    )
    num_question, num_user = dataset.interactions_shape()
    logger.info(f"Num question: {num_question}, num_user {num_user}.")
    if isinstance(user_features, pd.DataFrame):
        columns = USERS_NUM_COL
        if tags_columns:
            columns += tags_columns
        else:
            logger.warning(
                f"You will fit your dataset only on numerical columns for user features:\n {', '.join(columns)}"
            )
        logger.info("Fit partial dataset for users features")
        dataset.fit_partial(
            items=(users_id for users_id in user_features.id.tolist()),
            item_features=(pd.unique(user_features[columns].values.ravel('K'))),
        )
    if isinstance(question_features, pd.DataFrame):
        if not tags_columns:
            log_raise(
                logger=logger, err=ValueError("Arg tags_columns must be filled if question features want to be used ")
            )
        else:
            logger.info("Fit partial dataset for questions features")
            dataset.fit_partial(
                users=(question_id for question_id in question_features.question_id.tolist()),
                user_features=(pd.unique(question_features[tags_columns].values.ravel('K'))),
            )
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


def get_questions_features(data: pd.DataFrame, dataset: Dataset, tags_column: list[str]) -> COOMatrix:
    """Build questions features from lightfm dataset and dataframe

    Args:
        data: interactions to build
        dataset: light fm dataset

    Returns:
        COOMatrix: questions features matrix
    """
    logger.info("Building lightfm questions features")
    columns = ["question_id"] + tags_column
    return dataset.build_user_features(((row[0], list(row[1:])) for row in data[columns].itertuples(index=False, name=None)))  # type: ignore


def get_users_features(data: pd.DataFrame, dataset: Dataset, tags_column: list[str]) -> COOMatrix:
    """Build users features from lightfm dataset and dataframe

    Args:
        data: interactions to build
        dataset: light fm dataset

    Returns:
        COOMatrix: users features matrix
    """
    logger.info(f"Building lightfm users features")
    columns = ["id"] + tags_column + USERS_NUM_COL
    return dataset.build_item_features(((row[0], list(row[1:])) for row in data[columns].itertuples(index=False, name=None)))  # type: ignore
