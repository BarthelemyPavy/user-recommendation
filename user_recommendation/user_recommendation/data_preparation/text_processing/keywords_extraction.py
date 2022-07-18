"""Here is defined the main class to use to keep relevant words from textual data"""
from enum import Enum
from typing import Any, Callable, Optional, Union
import numpy as np
import pandas as pd
from functools import partial
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from keyphrase_vectorizers import KeyphraseCountVectorizer

import numpy.typing as npt

from user_recommendation.data_preparation.text_processing.tfidf import TfidfTransformerExtractor, Lemmatizer
from user_recommendation.data_preparation.text_processing.keybert import KeyBERTExtractor
from user_recommendation.errors import KeywordExtractorError
from user_recommendation.utils import log_raise
from user_recommendation import logger


class EKeywordExtractorTag(Enum):
    """Tags that can be used to choose Keyword extraction method

    Attributes:\
        TFIDF: Use tf-idf method\n
            (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)
        KEYBERT: Use keyBERT method (https://github.com/MaartenGr/KeyBERT)

    """

    TFIDF = "tfidf"
    KEYBERT = "keybert"


class KeywordsExtractor:
    """Class wrapping ways to extract keywords from text

    Attributes:

    """

    _model: Union[KeyBERTExtractor, Pipeline]
    _keyphrase_vectorizer: Optional[KeyphraseCountVectorizer] = None
    _feature_array: Optional[npt.NDArray[np.str_]] = None

    def __init__(self, extraction_method: EKeywordExtractorTag, **kwargs: str) -> None:
        """Class constructor, initialize the extractor method tu use

        Args:
            extraction_method: Tag to choose text extraction method to use
            kwargs: Define args to pass for extractor initialization:\n
                - For tfidf: see CountVectorizer from sklearn
                - For keybert:\n
                        model (str): Name of pre trained model (assuming we are using transformers models)
                                    See example to the official documentation

        """
        if extraction_method == EKeywordExtractorTag.TFIDF:
            self._model = Pipeline(
                [
                    ("lemma", Lemmatizer()),
                    ('vect', CountVectorizer(**kwargs)),
                    ('tfidf', TfidfTransformerExtractor()),
                ]
            )
        elif extraction_method == EKeywordExtractorTag.KEYBERT:
            self._model = KeyBERTExtractor(kwargs.get("model")) if "model" in kwargs else KeyBERTExtractor()
            self._keyphrase_vectorizer = KeyphraseCountVectorizer(pos_pattern="<N.*>", spacy_exclude=["parser", "ner"])

    def fit(self, data: Union[pd.Series[str], npt.NDArray[np.str_], list[str]]) -> None:
        """Fit method, require for tfidf extractor

        Args:
            data: Data to process
        """
        if isinstance(self._model, Pipeline):
            self._model.fit(data)
            self._feature_array = self._model["vect"].get_feature_names_out()

    @staticmethod
    def _partial_extract_keywords(  # type: ignore
        model: KeyBERTExtractor, vectorizer: KeyphraseCountVectorizer
    ) -> Callable[[Any], Any]:
        """Only for KeyBERTExtractor model. Use partial from functools to pre fill extract_keywords method

        Args:
            vectorizer (KeyphraseCountVectorizer): _description_

        Returns:
            Callable[[Any], Any]: _description_
        """
        return partial(model, vectorizer=vectorizer)

    def transform(
        self, data: Union[pd.Series[str], npt.NDArray[np.str_], list[str]], **kwargs: str
    ) -> Union[pd.Series[str], npt.NDArray[np.str_], list[str]]:
        """Take data to process in input and return best keywords as output

        Args:
            data: Data to process
            **kwargs: Define args to pass for extraction:\n
                - For tfidf:\n
                        top_n (int): Number of keyword to keep per document.See TfidfTransformerExtractor doc
                - For keybert:\n
                        docs: See parent class doc
                        candidates: See parent class doc. Defaults to None.
                        keyphrase_ngram_range: See parent class doc. Defaults to ....
                        stop_words[str, List: See parent class doc. Defaults to "english".
                        top_n: See parent class doc. Defaults to 5.
                        min_df: See parent class doc. Defaults to 1.
                        use_maxsum: See parent class doc. Defaults to False.
                        use_mmr: See parent class doc. Defaults to False.
                        diversity: See parent class doc. Defaults to 0.5.
                        nr_candidates: See parent class doc. Defaults to 20.
                        vectorizer: See parent class doc. Defaults to None.
                        highlight: See parent class doc. Defaults to False.
                        seed_keywords: See parent class doc. Defaults to None.
                        batch_size: Define a batch size if extraction need to be batch. Defaults to None.

        Returns:
            Union[pd.Series[str], npt.NDArray[np.str_], list[str]]: _description_
        """
        try:
            if isinstance(self._model, Pipeline):
                try:
                    check_is_fitted(self._model)
                except NotFittedError as error:
                    log_raise(
                        logger=logger,
                        err=KeywordExtractorError(
                            "Your extractor is not fitted, with tfidf is require to use .fit() method before extraction"
                        ),
                        original_err=error,
                    )
                if self._feature_array:
                    keywords = (
                        self._feature_array[self._model.transform(data, top_n=kwargs.get("top_n"))]
                        if "top_n" in kwargs
                        else self._feature_array[self._model.transform(data)]
                    )
                else:
                    log_raise(
                        logger=logger,
                        err=KeywordExtractorError(
                            "Error from tfidf extraction. Seems that feature array doesn't get from vectorizer"
                        ),
                    )

            elif isinstance(self._model, KeyBERTExtractor):
                extract_keywords = self._partial_extract_keywords(self._model, self._keyphrase_vectorizer)
                keywords = extract_keywords(data, **kwargs)
        except Exception as error:
            log_raise(
                logger=logger,
                err=KeywordExtractorError(
                    "Something went wrong with KeywordExtractor. Please check traceback for more information."
                ),
                original_err=error,
            )
        return keywords


def get_tfidf_pipeline(stop_words: str, strip_accents: str) -> Pipeline:
    """Create and return sklearn pipeline

    Args:
        stop_words (str): _description_
        strip_accents (str): _description_

    Returns:
        Pipeline: _description_
    """
    return Pipeline(
        [
            ("lemma", Lemmatizer()),
            ('vect', CountVectorizer(stop_words=stop_words, strip_accents=strip_accents)),
            ('tfidf', TfidfTransformerExtractor()),
        ]
    )
