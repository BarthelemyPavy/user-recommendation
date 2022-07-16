"""Train, Test, Val generation"""
import math
from typing import Tuple
import numpy as np
from user_recommendation.data_preparation.split_datasets import Datasets, Split
import pandas as pd

from user_recommendation.errors import MissingAttribute
from user_recommendation.utils import log_raise, log_attribute_per_dataset
from user_recommendation import logger


class PreprocessDataframes:
    """Apply transformations from input data to get a readt to use dataset for train test val split

    Attributes:
        _dataframe_to_split: dataframe containing all information used to split
    """

    _dataframe_to_split: pd.Dataframe

    def __init__(self, questions: pd.DataFrame, answers: pd.DataFrame, users: pd.DataFrame) -> None:
        """Class constructor

        Args:
            questions: DataFrame containing questions
            answers: DataFrame containing mapping between questions, answers and users
            users: DataFrame containing users
        """
        self._questions = questions
        self._answer = answers
        self._users = users
        self._dataframe_to_split = self._answer[["answer_id", "user_id", "question_id", "score", "date"]]

    @property
    def dataframe_to_split(self) -> pd.DataFrame:
        """Get _dataframe_to_split attribute

        Returns:
            pd.DataFrame:
        """
        return self._dataframe_to_split

    def _is_accepted_answer(self, new_col: str = "is_accepted_answer") -> None:
        """Define a boolean series to identify if a question has an accepted answer

        Args:
            new_col: name of the new column
        """
        self._dataframe_to_split = self._dataframe_to_split.merge(
            self._questions[["accepted_answer_id"]], how="left", left_on="answer_id", right_on="accepted_answer_id"
        )
        self._dataframe_to_split[new_col] = np.where(self._dataframe_to_split.accepted_answer_id.isna(), 0, 1)

    def _compute_number_of_answers_per_question(self, new_col: str = "number_answers_per_question") -> None:
        """Compute number of answers per question

        Args:
            new_col: name of the new column
        """
        self._dataframe_to_split = self._dataframe_to_split.merge(
            self._answer[["question_id", "answer_id"]]
            .groupby("question_id")
            .count()
            .reset_index()
            .rename(columns={"answer_id": new_col}),
            on="question_id",
        )

    def _get_positive_interactions(self, threshold: int, take_accepted_answer: bool) -> Tuple[list[int], list[int]]:
        """Filter a dataframe based on threshold apply on column

        Args:
            threshold: Threshold to apply
            column: Filtering column
            take_accepted_answer: Define is we take answer with column < threshold but answer accepted. default True

        Raises:
            MissingAttribute: Raised if a datafram column is missing

        Returns:
            Tuple[list[int], list[int]]: Tuple containing positive interactions list and negative interactions list
        """
        try:
            if take_accepted_answer:

                return_pos = self._dataframe_to_split[
                    (self._dataframe_to_split.score >= threshold) | (self._dataframe_to_split.is_accepted_answer == 1)
                ].answer_id.tolist()

            else:
                return_pos = self._dataframe_to_split[self._dataframe_to_split.score >= threshold].answer_id.tolist()
            return_neg = self._dataframe_to_split[
                ~self._dataframe_to_split.answer_id.isin(return_pos)
            ].answer_id.tolist()
        except KeyError as err:
            # score and answer_id are assigned at the initialization,
            # if a key error raised means that is_accepted_answer is missing
            log_raise(
                logger=logger,
                err=MissingAttribute(attribute="is_accepted_answer", valid_attributes=self._dataframe_to_split.columns),
                original_err=err,
            )
        return (return_pos, return_neg)

    def _metric_to_classes(
        self, metric_name: str, high_percentile: float = 0.99, medium_percentile: float = 0.7
    ) -> pd.Series:
        """Convert a metric to classes: High, Medium, Low

        Args:
            metric_name: Metric to label
            high_percentile: Percentile for high label
            medium_percentile: Percentile for medium label

        Raises:
            MissingAttribute: Raised if a datafram column is missing

        Returns:
            pd.Series: labeled metric
        """
        if 0 > high_percentile > 1 or 0 > medium_percentile < 1:
            log_raise(logger=logger, err=ValueError("high_percentile and medium_percentile must be between 0 and 1."))
        if metric_name not in self._dataframe_to_split.columns:
            log_raise(
                logger=logger,
                err=MissingAttribute(attribute=metric_name, valid_attributes=self._dataframe_to_split.columns),
            )
        high_threshold = self._dataframe_to_split.drop_duplicates(subset="question_id")[metric_name].quantile(
            high_percentile
        )
        medium_threshold = self._dataframe_to_split.drop_duplicates(subset="question_id")[metric_name].quantile(
            medium_percentile
        )

        return np.where(
            self._dataframe_to_split[metric_name] >= high_threshold,
            "High",
            np.where(self._dataframe_to_split[metric_name] >= medium_threshold, "Medium", "Low"),
        )

    def execute(self, positive_interactions_threshold: int, take_accepted_answer: bool = True) -> pd.DataFrame:
        """Main function of the class

        Returns:
            pd.DataFrame: final dataframe to split
        """
        logger.info("Preprocess input Data")
        nb_answer_column = "number_answers_per_question"
        self._is_accepted_answer()
        self._compute_number_of_answers_per_question(new_col=nb_answer_column)
        positive_answer_ids, _ = self._get_positive_interactions(
            threshold=positive_interactions_threshold, take_accepted_answer=take_accepted_answer
        )
        self._dataframe_to_split = self._dataframe_to_split.loc[
            self._dataframe_to_split.answer_id.isin(positive_answer_ids)
        ]
        self._dataframe_to_split["question_label"] = self._metric_to_classes(metric_name=nb_answer_column)
        logger.info("Preprocess Done")
        return self._dataframe_to_split


class ColdStartSplit(Split):
    """Cold Start split from preprocessed dataframe"""

    def execute(  # type: ignore
        self, test_val_size: float, split_size: float, random_state: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main function of the class

        Args:
            test_val_size: Rate of all questions to be used for test and val datasets
            split_size: Rate of split for test_val datasets
            random_state: random_state for split reproducibility

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: tuple of dataframes containing test dataframe and val dataframe,
                                                (test_df, val_df)
        """
        data_type = "cold_start"
        questions = self._deduplicate_questions()
        test_ids, val_ids = self._test_val_split(
            unique_questions=questions,
            first_split_size=test_val_size * split_size,
            random_state=random_state,
        )
        test_df = self._preprocessed_df[self._preprocessed_df.question_id.isin(test_ids)]
        val_df = self._preprocessed_df[self._preprocessed_df.question_id.isin(val_ids)]

        test_df["data_type"] = data_type
        val_df["data_type"] = data_type

        log_attribute_per_dataset(test_df, "question_label", logger=logger, desc="test cold start dataframe")
        log_attribute_per_dataset(val_df, "question_label", logger=logger, desc="val cold start dataframe")
        return (test_df, val_df)


class WarmStartSplit(Split):
    """Warm Start split from preprocessed dataframe

    Attributes:
        _cold_start_len: number of ids used for cold start
    """

    def __init__(self, preprocessed_df: pd.DataFrame, cold_start_ids: list[int]) -> None:
        """Overload constructor

        Args:
            preprocessed_df: See parent class docs
            cold_start_ids: Question ids used for cold start split
        """
        super().__init__(preprocessed_df)
        self._preprocessed_df = self._preprocessed_df[~self._preprocessed_df.question_id.isin(cold_start_ids)]
        self._cold_start_len = len(cold_start_ids)

    def _leave_last_item_strategy(self, dataframe: pd.DataFrame, rate_last_item: float) -> pd.DataFrame:
        """Define the strategy of test train split for warm start,
                i.e the question has been seen in the training dataset.

        Leave One Last Item Strategy from https://arxiv.org/pdf/2007.13237.pdf

        Args:
            dataframe: dataframe to filter
            rate_last_item: rate of last item to keep

        Returns:
            pd.DataFrame: Filtered dataframe
        """
        dataframe_group = dataframe.groupby("question_id")
        return dataframe_group.apply(
            lambda x: x.sort_values(by="date", ascending=False).head(math.ceil(x.size * rate_last_item))
        ).reset_index(drop=True)

    def execute(self, split_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:  # type: ignore
        """Main function of the class

        Args:
            split_size: Rate of split for test_val datasets
            random_state: random_state for split reproducibility

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: tuple of dataframes containing test dataframe and val dataframe,
                                                (test_df, val_df)
        """
        data_type = "warm_start"
        rate_last_item = 0.3
        # Compute number of questions to take for warm start
        warm_start_test_nb = int((1 - split_size) * self._cold_start_len / split_size)
        questions = self._deduplicate_questions()
        test_ids, val_ids = self._test_val_split(
            unique_questions=questions,
            first_split_size=warm_start_test_nb,
            random_state=random_state,
        )
        test_df = self._leave_last_item_strategy(
            dataframe=self._preprocessed_df[self._preprocessed_df.question_id.isin(test_ids)],
            rate_last_item=rate_last_item,
        )
        val_df = self._leave_last_item_strategy(
            dataframe=self._preprocessed_df[self._preprocessed_df.question_id.isin(val_ids)],
            rate_last_item=rate_last_item,
        )
        test_df["data_type"] = data_type
        val_df["data_type"] = data_type

        log_attribute_per_dataset(test_df, "question_label", logger=logger, desc="test cold start dataframe")
        log_attribute_per_dataset(val_df, "question_label", logger=logger, desc="val cold start dataframe")
        return (test_df, val_df)


class CreateDatasetsObject:
    """Create Datasets object containing train test and val dataframes"""

    def __init__(self, preprocessed_df: pd.DataFrame) -> None:
        """Class constructor

        Args:
            preprocessed_df: DataFrame preprocessed by previous step, see PreprocessDataframes for more information
        """
        self._preprocessed_df = preprocessed_df
        self._datasets = Datasets(training=None, test=None, validation=None)

    def _build_train_df(self, test_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
        """Build train dataframe from test and validation.

        Args:
            test_df (pd.DataFrame): test dataframe
            val_df (pd.DataFrame): validation dataframe

        Returns:
            pd.DataFrame: train dataframe
        """
        answer_ids = test_df.answer_id.tolist() + val_df.answer_id.tolist()
        return self._preprocessed_df[~self._preprocessed_df.answer_id.isin(answer_ids)]

    def execute(
        self, test_cs: pd.DataFrame, test_ws: pd.DataFrame, val_cs: pd.DataFrame, val_ws: pd.DataFrame
    ) -> Datasets:
        """Take multiple dataframe and create

        Args:
            test_cs: Cold Start test
            test_ws: Warm Start test
            val_cs: Cold Start val
            val_ws: Warm Start val

        Returns:
            Datasets: datasets object containing train test val dataframes
        """
        self._datasets.test = pd.concat([test_cs, test_ws], ignore_index=True)
        self._datasets.validation = pd.concat([val_cs, val_ws], ignore_index=True)
        self._datasets.training = self._build_train_df(test_df=self._datasets.test, val_df=self._datasets.validation)
        return self._datasets
