"""Some utils functions"""

from pathlib import Path
from typing import Iterator, List, Tuple, Union
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


def parse_html_tags(docs: str) -> str:
    """Get a doc containing html tags and remove it

    Args:
        text: Unclean doc containing html tags

    Returns:
        str: Clean doc without html tags
    """
    to_return = docs
    if isinstance(docs, str):
        to_return = BeautifulSoup(docs, "lxml").text
    return to_return


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
