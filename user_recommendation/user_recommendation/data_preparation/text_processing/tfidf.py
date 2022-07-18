"""Use tf-idf to process text and extract relevant information for cold start recommendations"""
from __future__ import annotations
from typing import Optional, TypeVar
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

import numpy.typing as npt

SklearnType = TypeVar("SklearnType")


class Lemmatizer(BaseEstimator, TransformerMixin):
    """Define a lemmatizer to use into sklearn Pipeline"""

    def __init__(self) -> None:
        """Init nltk Lemmatizer"""
        self.lemma = WordNetLemmatizer()

    def fit(self, X: SklearnType, y: Optional[SklearnType] = None) -> Lemmatizer:
        """Nothing happens here

        Args:
            X:
            y: Defaults to None.

        Returns:
            Lemmatizer:
        """
        return self

    def _lemmatize(self, text: str) -> str:
        """Take a string and lemmatize it

        Args:
            text: Input string

        Returns:
            str: Lemmatized string
        """
        return " ".join([self.lemma.lemmatize(word) for word in word_tokenize(text)])

    def transform(self, X: list[str], y: Optional[SklearnType] = None) -> list[str]:
        """Take a list of string in input and lemmatized each of them

        Args:
            X: list of string to process
            y: Defaults to None.

        Returns:
            list[str]: list of string lemmatized
        """
        return [self._lemmatize(text) for text in X]


class TfidfTransformerExtractor(TfidfTransformer):
    """Inherit from TfidfTransformer"""

    def transform(self, X: SklearnType, top_n: int = 5, copy: bool = True) -> npt.NDArray[np.int_]:
        """Override transform method to keep only top n keywords

        Args:
            X: See parent doc
            top_n: Number of keyword to keep per document. Defaults to 5.
            copy: See parent doc. Defaults to True.

        Returns:
            npt.NDArray[np.int_]: array order and filter
        """
        sparse_matrix = super().transform(X, copy)
        tfidf_sorting = np.apply_along_axis(
            lambda x: np.argsort(x)[-min(top_n, x.size) : :], 1, sparse_matrix.toarray()
        )
        return tfidf_sorting
