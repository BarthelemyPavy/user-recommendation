"""Use KeyBERT to process text and extract relevant information for cold start recommendations"""
from typing import Iterator, List, Optional, Tuple, Union
from tqdm import tqdm
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer


class KeyBERTExtractor(KeyBERT):
    """Extraction class inherited from KeyBERT"""

    _batch_size: Optional[int]

    @staticmethod
    def batch_generator(docs: list[str], batch_size: int) -> Iterator[list[str]]:
        """Generate batch to process

        Args:
            docs: List of doc to process
            batch_size: Size of batch to generate

        Yields:
            Iterator[list[str]]: Generated batch
        """
        docs_len = len(docs)
        for idx in tqdm(range(0, docs_len, batch_size)):
            yield docs[idx : min(idx + batch_size, docs_len)]

    @staticmethod
    def _tuple_to_str(data: list[Tuple[str, float]]) -> list[str]:
        """Re format output of KeyBERT extract_keywords method

        Args:
            data: data to re-format

        Returns:
            list[str]: output as ['keyword1', 'keyword2', ..]
        """
        return [tup[0] for tup in data]

    def extract_keywords(
        self,
        docs: Union[str, List[str]],
        candidates: Optional[List[str]] = None,
        keyphrase_ngram_range: Tuple[int, int] = (1, 1),
        stop_words: Union[str, List[str]] = "english",
        top_n: int = 5,
        min_df: int = 1,
        use_maxsum: bool = False,
        use_mmr: bool = False,
        diversity: float = 0.5,
        nr_candidates: int = 20,
        vectorizer: CountVectorizer = None,
        highlight: bool = False,
        seed_keywords: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
    ) -> Union[List[str], List[List[str]]]:
        """Add the possibility to batch extracting keywords

        Args:
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
            Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]: See parent class doc
        """
        self._batch_size = batch_size if batch_size else None
        keywords = super().extract_keywords(
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
        reformat_keywords: Union[List[str], List[List[str]]] = (
            [self._tuple_to_str(tup_list) for tup_list in keywords]
            if isinstance(keywords[0], list)
            else self._tuple_to_str(keywords)
        )
        return reformat_keywords

    def _extract_keywords_multiple_docs(
        self,
        docs: List[str],
        keyphrase_ngram_range: Tuple[int, int] = ...,  # type: ignore
        stop_words: str = "english",
        top_n: int = 5,
        min_df: int = 1,
        vectorizer: CountVectorizer = None,
    ) -> List[List[Tuple[str, float]]]:
        """Batch keyword extraction if batch_size is not None

        Args:
            docs: See parent class doc
            keyphrase_ngram_range: See parent class doc. Defaults to ....
            stop_words: See parent class doc. Defaults to "english".
            top_n: See parent class doc. Defaults to 5.
            min_df: See parent class doc. Defaults to 1.
            vectorizer: See parent class doc. Defaults to None.

        Returns:
            List[List[Tuple[str, float]]]: See parent class doc
        """
        if self._batch_size:
            keywords = []
            for docs_batch in self.batch_generator(docs, self._batch_size):
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
