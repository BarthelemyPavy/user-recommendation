"""Test text cleaning functions"""
from unittest import TestCase

from user_recommendation.data_preparation.text_processing.tfidf import Lemmatizer, EStemTag


class TextCleaningTest(TestCase):
    """Unittest on text cleaning function"""

    def setUp(self) -> None:
        self.lemma = Lemmatizer(EStemTag.STEMMER)

    def test_remove_url(self) -> None:
        """Test function string_to_enum"""

        string_to_clean = "https://docs.metaflow.org/metaflow/tagging"
        expected_string = ""
        self.assertTrue(isinstance(self.lemma.remove_url_and_number(string_to_clean), str))
        self.assertEqual(self.lemma.remove_url_and_number(string_to_clean), expected_string)

        string_to_clean = "https://docs.metaflow.org/metaflow/tagging and some other text"
        expected_string = " and some other text"
        self.assertEqual(self.lemma.remove_url_and_number(string_to_clean), expected_string)

        string_to_clean = "and some other text https://docs.metaflow.org/metaflow/tagging"
        expected_string = "and some other text "
        self.assertEqual(self.lemma.remove_url_and_number(string_to_clean), expected_string)

    def test_remove_numbers(self) -> None:
        """Test function string_to_enum"""

        string_to_clean = "126584"
        expected_string = ""
        self.assertTrue(isinstance(self.lemma.remove_url_and_number(string_to_clean), str))
        self.assertEqual(self.lemma.remove_url_and_number(string_to_clean), expected_string)

        string_to_clean = "56985 and some other55 text"
        expected_string = " and some other text"
        self.assertEqual(self.lemma.remove_url_and_number(string_to_clean), expected_string)

        string_to_clean = "a5nd some other55 text6666"
        expected_string = "and some other text"
        self.assertEqual(self.lemma.remove_url_and_number(string_to_clean), expected_string)
