"""Use tf-idf to process text and extract relevant information for cold start recommendations"""
from __future__ import annotations
from typing import Iterator, Optional, TypeVar
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
