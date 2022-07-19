"""Here is defined the main class to use to keep relevant words from textual data"""
from enum import Enum
from typing import Optional, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from keyphrase_vectorizers import KeyphraseCountVectorizer

import numpy.typing as npt

from user_recommendation.data_preparation.text_processing.tfidf import TfidfTransformerExtractor, Lemmatizer, EStemTag
from user_recommendation.data_preparation.text_processing.keybert import KeyBERTExtractor
from user_recommendation.errors import KeywordExtractorError, InvalidTag
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
                - For tfidf:\n
                        top_n: number of tag to keep per document
                        stem: choose between stemmer or lemmatizer as preprocesser
                        CountVectorizer args: see CountVectorizer from sklearn
                - For keybert:\n
                        model (str): Name of pre trained model (assuming we are using transformers models)
                                    See example to the official documentation

        """
        if extraction_method == EKeywordExtractorTag.TFIDF:
            top_n: int = kwargs.get("top_n", 5)  # type: ignore
            kwargs.pop("top_n", None)
            stem: EStemTag = kwargs.get("stem", EStemTag.STEMMER)  # type: ignore
            kwargs.pop("stem", None)
            self._model = Pipeline(
                [
                    (stem.value, Lemmatizer(stem=stem)),
                    ('vect', CountVectorizer(**kwargs)),
                    ('tfidf', TfidfTransformerExtractor(top_n=top_n)),
                ]
            )
        elif extraction_method == EKeywordExtractorTag.KEYBERT:
            self._model = KeyBERTExtractor(kwargs.get("model")) if "model" in kwargs else KeyBERTExtractor()
            self._keyphrase_vectorizer = KeyphraseCountVectorizer(
                pos_pattern="<N.*>",
                spacy_exclude=["parser", "ner"],
            )
        else:
            log_raise(
                logger=logger,
                err=KeywordExtractorError(f"Extractor {extraction_method.value}"),
                original_err=InvalidTag(stem.value, EStemTag.__doc__),  # type: ignore
            )

    def fit(self, data: Union[pd.Series, npt.NDArray[np.str_], list[str]]) -> None:
        """Fit method, require for tfidf extractor

        Args:
            data: Data to process
        """
        logger.info("Fit of KeywordsExtractor is running")
        if isinstance(self._model, Pipeline):
            self._model.fit(data)
            self._feature_array = self._model["vect"].get_feature_names_out()
        logger.info("Fit Done")

    def transform(
        self, data: Union[str, list[str]], **kwargs: str
    ) -> Union[pd.Series, npt.NDArray[np.str_], list[str]]:
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
            Union[pd.Series, npt.NDArray[np.str_], list[str]]: _description_
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
                if not isinstance(self._feature_array, np.ndarray):
                    log_raise(
                        logger=logger,
                        err=KeywordExtractorError(
                            "Error from tfidf extraction. Seems that feature array doesn't get from vectorizer"
                        ),
                    )
                else:
                    keywords = self._feature_array[self._model.transform(data)]

            elif isinstance(self._model, KeyBERTExtractor):
                keywords = self._model.extract_keywords(docs=data, vectorizer=self._keyphrase_vectorizer)
        except Exception as error:
            log_raise(
                logger=logger,
                err=KeywordExtractorError(
                    "Something went wrong with KeywordExtractor. Please check traceback for more information."
                ),
                original_err=error,
            )
        return keywords
