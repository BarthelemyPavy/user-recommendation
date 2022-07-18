"""Use tf-idf to process text and extract relevant information for cold start recommendations"""
from __future__ import annotations
from enum import Enum
import re
from typing import Optional, TypeVar
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer

import numpy.typing as npt

from user_recommendation.utils import log_raise
from user_recommendation.errors import InvalidTag
from user_recommendation import logger

SklearnType = TypeVar("SklearnType")


class EStemTag(Enum):
    """Tags that can be used to choose between stemmer or lemmatizer

    Attributes:\
        STEMMER: Use SnowballStemmer from nltk
        LEMMATIZER: Use WordNetLemmatizer from nltk

    """

    STEMMER = "stemmer"
    LEMMATIZER = "lemmatizer"


class Lemmatizer(BaseEstimator, TransformerMixin):
    """Define a lemmatizer to use into sklearn Pipeline"""

    def __init__(self, stem: EStemTag) -> None:
        """Init nltk Lemmatizer

        Args:
            stem: use stemmer or lemmatizer
        """
        if stem == EStemTag.LEMMATIZER:
            self.lemma = WordNetLemmatizer()
        elif stem == EStemTag.STEMMER:
            self.lemma = SnowballStemmer("english")
        else:
            log_raise(logger=logger, err=InvalidTag(stem.value, EStemTag.__doc__))  # type: ignore

    def fit(self, X: SklearnType, y: Optional[SklearnType] = None) -> Lemmatizer:
        """Nothing happens here

        Args:
            X:
            y: Defaults to None.

        Returns:
            Lemmatizer:
        """
        return self

    def _clean_text(self, text: str) -> str:
        """Get an noisy text in input and remove html tags and url

        Args:
            text: Unclean input text

        Returns:
            str: Clean text
        """
        text = self.remove_url_and_number(text)
        text = self.parse_html_tags(text)

        return text

    @staticmethod
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

    @staticmethod
    def remove_url_and_number(text: str) -> str:
        """Get a text containing url or number and remove it

        Args:
            text: Unclean text containing url or number

        Returns:
            str: Clean text without url and number
        """
        to_return = text
        if isinstance(text, str):
            text_without_ulr = re.sub(r"https?://[A-Za-z0-9./]+", "", text, flags=re.MULTILINE)
            to_return = re.sub("\d+", "", text_without_ulr, flags=re.MULTILINE)
        return to_return

    def _lemmatize(self, text: str) -> str:
        """Take a string and lemmatize it

        Args:
            text: Input string

        Returns:
            str: Lemmatized string
        """
        if isinstance(self.lemma, WordNetLemmatizer):
            to_return = " ".join([self.lemma.lemmatize(word) for word in word_tokenize(text)])
        elif isinstance(self.lemma, SnowballStemmer):
            to_return = " ".join([self.lemma.stem(word) for word in word_tokenize(text)])
        return to_return

    def transform(self, X: list[str], y: Optional[SklearnType] = None) -> list[str]:
        """Take a list of string in input and lemmatized each of them

        Args:
            X: list of string to process
            y: Defaults to None.

        Returns:
            list[str]: list of string lemmatized
        """
        return [self._lemmatize(self._clean_text(text)) for text in X]


class TfidfTransformerExtractor(TfidfTransformer):
    """Inherit from TfidfTransformer"""

    _batch_size: int = 256

    def __init__(
        self,
        *,
        norm: str = "l2",
        use_idf: bool = True,
        smooth_idf: bool = True,
        sublinear_tf: bool = False,
        top_n: int = 5,
    ) -> None:
        """_summary_

        Args:
            norm: _description_. Defaults to "l2".
            use_idf: _description_. Defaults to True.
            smooth_idf: _description_. Defaults to True.
            sublinear_tf: _description_. Defaults to False.
            top_n: Number of keyword to keep per document. Defaults to 5.
        """
        super().__init__(norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
        self._top_n = top_n

    def transform(self, X: SklearnType, copy: bool = True) -> npt.NDArray[np.int_]:
        """Override transform method to keep only top n keywords

        Args:
            X: See parent doc
            copy: See parent doc. Defaults to True.

        Returns:
            npt.NDArray[np.int_]: array order and filter
        """
        sparse_matrix = super().transform(X, copy)
        sparse_matrix = sparse_matrix.toarray()

        tfidf_order = np.apply_along_axis(lambda x: np.argsort(x)[-min(self._top_n, x.size) : :], 1, sparse_matrix)
        return tfidf_order
