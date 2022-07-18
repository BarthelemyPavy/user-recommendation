"""Test text cleaning funstions"""
from unittest import TestCase
from sklearn.feature_extraction.text import CountVectorizer

from user_recommendation.data_preparation.text_processing.keywords_extraction import (
    KeywordsExtractor,
    EKeywordExtractorTag,
)
from user_recommendation.errors import KeywordExtractorError
from user_recommendation.data_preparation.text_processing.tfidf import TfidfTransformerExtractor


class KeywordsExtractorTest(TestCase):
    """Unittest on keyword extractor class"""

    def test_not_fitted(self) -> None:
        """Test function string_to_enum"""
        extractor = KeywordsExtractor(extraction_method=EKeywordExtractorTag.TFIDF)
        with self.assertRaises(KeywordExtractorError):
            extractor.transform(["test to process"])


class TfidfTransformerExtractorTest(TestCase):
    """Unittest on TfidfTransformerExtractor class"""

    def test_top_n(self) -> None:
        """Test function string_to_enum"""
        vectorizer = CountVectorizer()
        vectorized = vectorizer.fit_transform(
            [
                """Lorem Ipsum is simply dummy text of the printing and typesetting industry.
                             Lorem Ipsum has been the industry's standard dummy text ever since the 1500s,
                             when an unknown printer took a galley of type and scrambled it to make a type
                             specimen book. It has survived not only five centuries, but also the leap into
                             electronic typesetting, remaining essentially unchanged. It was popularised in the
                             1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more
                             recently with desktop publishing software like Aldus PageMaker including versions of
                             Lorem Ipsum."""
            ]
        )
        for i in range(1, 6):
            extractor = TfidfTransformerExtractor(top_n=i)
            extractor.fit(vectorized)
            output = extractor.transform(vectorized)
            self.assertTrue(output.shape, (1, i))
