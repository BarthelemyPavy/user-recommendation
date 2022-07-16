"""Some utils functions"""

from enum import Enum
import json
import logging
from pathlib import Path
from dataclasses import dataclass
import pickle
import sys
import traceback
from typing import Iterator, List, Optional, Tuple, TypeVar, Union
import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

DATA_DIRECTORY_PATH = Path().absolute().parents[0] / "data"
AVAILABLE_FILES = ["answers.json", "questions.json", "users.json"]


def read_data(file_name: str) -> pd.DataFrame:
    """Read json file

    Args:
        file_name: Name of the file to read

    Returns:
        pd.DataFrame: DataFrame containing data from json
    """
    if file_name not in AVAILABLE_FILES:
        raise KeyError(f"File {file_name} doesn't exists\n Available files are: {''.join(AVAILABLE_FILES)}")

    return pd.read_json(DATA_DIRECTORY_PATH / Path(file_name), lines=True)


def parse_html_tags(text: str) -> str:
    """Get a text containing html tags and remove it

    Args:
        text: Unclean text containing html tags

    Returns:
        str: Clean text without html tags
    """
    to_return = text
    if isinstance(text, str):
        to_return = BeautifulSoup(text, "lxml").text
    return to_return


def word_cloud_generation(texts: list[str], title: str, max_words: int) -> None:
    """Generate a word cloud from texts

    Args:
        texts: Texts to parse to identify most represented words
        title: Title of the image
        max_words: Max number of words to take care
    """
    wc = WordCloud(background_color="black", max_words=max_words, stopwords=STOPWORDS)
    wc.generate(" ".join(texts))
    plt.figure(figsize=(8, 8))
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.title(title, fontsize=20)
    plt.imshow(wc.recolor(colormap='gist_earth', random_state=244), alpha=0.98)


# Text processing
class KeywordExtractor(KeyBERT):
    """Extraction class inherited from KeyBERT"""

    _batch_size: int

    @classmethod
    def _batch_generator(cls, docs: list[str], batch_size: int) -> Iterator[str]:
        """Generate batch to process

        Args:
            docs: List of doc to process
            batch_size: Size of batch to generate

        Yields:
            Iterator[str]: Generated batch
        """
        docs_len = len(docs)
        for idx in range(0, docs_len, batch_size):
            yield docs[idx : min(idx + batch_size, docs_len)]

    def extract_keywords(
        self,
        docs: Union[str, List[str]],
        candidates: List[str] = None,
        keyphrase_ngram_range: Tuple[int, int] = ...,
        stop_words: Union[str, List[str]] = "english",
        top_n: int = 5,
        min_df: int = 1,
        use_maxsum: bool = False,
        use_mmr: bool = False,
        diversity: float = 0.5,
        nr_candidates: int = 20,
        vectorizer: CountVectorizer = None,
        highlight: bool = False,
        seed_keywords: List[str] = None,
        batch_size: int = None,
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """Add the possibility to batch extracting keywords

        Args:
            batch_size: Define a batch size if extraction need to be batch. Defaults to None.
        """
        self._batch_size = batch_size
        return super().extract_keywords(
            docs,
            candidates,
            keyphrase_ngram_range,
            stop_words,
            top_n,
            min_df,
            use_maxsum,
            use_mmr,
            diversity,
            nr_candidates,
            vectorizer,
            highlight,
            seed_keywords,
        )

    def _extract_keywords_multiple_docs(
        self,
        docs: List[str],
        keyphrase_ngram_range: Tuple[int, int] = ...,
        stop_words: str = "english",
        top_n: int = 5,
        min_df: int = 1,
        vectorizer: CountVectorizer = None,
    ) -> List[List[Tuple[str, float]]]:
        """Batch keyword extraction if batch_size is not None"""
        if self._batch_size:
            keywords = []
            for docs_batch in self._batch_generator(docs, self._batch_size):
                keywords.extend(
                    super()._extract_keywords_multiple_docs(
                        docs_batch, keyphrase_ngram_range, stop_words, top_n, min_df, vectorizer
                    )
                )
        else:
            keywords = super()._extract_keywords_multiple_docs(
                docs, keyphrase_ngram_range, stop_words, top_n, min_df, vectorizer
            )
        return keywords


class TfidfTransformerExtractor(TfidfTransformer):
    def transform(self, X, top_n=5, copy=True):
        sparse_matrix = super().transform(X, copy)
        tfidf_sorting = np.apply_along_axis(
            lambda x: np.argsort(x)[-min(top_n, x.size) : :], 1, sparse_matrix.toarray()
        )
        return tfidf_sorting


# Train Test Eval generation
def get_positive_interactions(
    dataframe: pd.DataFrame, threshold: int, column: str, take_accepted_answer: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter a dataframe based on threshold apply on column

    Args:
        dataframe: Dataframe to filter
        threshold: Threshold to apply
        column: Filtering column
        take_accepted_answer: Define is we take answer with column < threshold but answer accepted. default True

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing positive interactions dataframe and negative interactions dataframe
    """
    if take_accepted_answer:
        df_return_pos = dataframe[(dataframe[column] >= threshold) | (dataframe.is_accepted_answer == 1)]
    else:
        df_return_pos = dataframe[dataframe[column] >= threshold]
    df_return_neg = dataframe[~dataframe.index.isin(df_return_pos.index)]
    return (df_return_pos, df_return_neg)


def metric_to_classes(metric: pd.Series, high_percentile: float, medium_percentile: float) -> pd.Series:
    """Convert a metric to classes: High, Medium, Low

    Args:
        metric: Metric to label
        high_percentile: Percentile for high label
        medium_percentile: Percentile for medium label

    Returns:
        pd.Series: labeled metric
    """
    if 0 > high_percentile > 1 or 0 > medium_percentile < 1:
        raise ValueError("high_percentile and medium_percentile must be between 0 and 1.")
    high_threshold = metric.quantile(high_percentile)
    medium_threshold = metric.quantile(medium_percentile)

    return np.where(metric >= high_threshold, "High", np.where(metric >= medium_threshold, "Medium", "Low"))


@dataclass
class DataArtifacts:
    """Paths to artifacts use for prediciton

    Attributes:\
        path_to_train_dataset: path to train dataset
        path_to_test_warm_start_dataset: path to test warm_start dataset
        path_to_test_cold_start_dataset: path to test cold_start dataset
        path_to_eval_warm_start_dataset: path to eval warm_start dataset
        path_to_eval_cold_start_dataset: path to eval cold_start dataset
    """

    path_to_train_dataset: str
    path_to_test_warm_start_dataset: str
    path_to_test_cold_start_dataset: str
    path_to_eval_warm_start_dataset: str
    path_to_eval_cold_start_dataset: str

    def asdict(self) -> dict[str, str]:
        """Convert to dictionary

        Returns:\
            dict[str, str]: DataArtifacts attribute as dictionary
        """
        return vars(self)


SERIALIZED_DATA_ARTIFACT_NAMES = DataArtifacts(
    path_to_train_dataset="data/datasets/train.csv",
    path_to_test_warm_start_dataset="data/datasets/test_warm_start.csv",
    path_to_test_cold_start_dataset="data/datasets/test_cold_start.csv",
    path_to_eval_warm_start_dataset="data/datasets/eval_warm_start.csv",
    path_to_eval_cold_start_dataset="data/datasets/eval_cold_start.csv",
)


class ESerializerExtension(Enum):
    """List of extension handle by serializer

    Attributes
        CSV: csv extension
        PKL: pickle extension

    """

    CSV = "csv"
    PKL = "pkl"
    JSON = "json"


T = TypeVar("T")


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
        self._logger = log

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
        with open(self._path, "wb") as output_file:
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


# The format string to be used by the logger across projects.
FORMAT: str = "{asctime} :: {funcName} :: {levelname} :: {message}"


def _init(name: Optional[str] = None) -> logging.Logger:
    """Initialize the logger for the application using parameters from environment."""
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()

    level: int = logging.DEBUG
    formatter: logging.Formatter = logging.Formatter(FORMAT, style="{")

    handler.setFormatter(formatter)
    handler.setStream(sys.stderr)

    logger.addHandler(handler)
    logger.setLevel(level)

    return logger


log = _init()
"""A default logger instance for the application."""


def log_raise(
    logger: logging.Logger,
    err: Exception,
    original_err: Optional[Exception] = None,
) -> None:
    """Log an error and raise exception

    Args:
        logger: logger instance
        err: custom exception thrown by cool job
        original_err: original exception custom exception wraps around, if any
            (default None)

    Raises:
        Exception: systematically
    """
    logger.error(msg=str(err), trace=traceback.format_exc())
    if original_err:
        raise err from original_err
    raise err
