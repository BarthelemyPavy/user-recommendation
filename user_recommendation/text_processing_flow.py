"""File where text processing Flow is defined"""
from pathlib import Path
from metaflow import FlowSpec, step, Parameter
import numpy as np
import numpy.typing as npt
import pandas as pd
from user_recommendation import logger
from user_recommendation.data_preparation.text_processing.keywords_extraction import KeywordsExtractor


class TextProcessingFlow(FlowSpec):
    """Flow used to extract keywords from textual data"""

    def _batch_tfidf(self, extractor: KeywordsExtractor, data_to_batch: list[str]) -> npt.NDArray[np.str_]:
        """_summary_

        Args:
            extractor (KeywordsExtractor): _description_

        Returns:
            npt.NDArray[np.str_]: _description_
        """
        from user_recommendation.data_preparation.text_processing.keybert import KeyBERTExtractor

        keywords = []
        for batch in KeyBERTExtractor.batch_generator(data_to_batch, self.config.get("batch_size")):
            keywords.extend(extractor.transform(batch).tolist())  # type: ignore
        return np.array(keywords)

    random_state = Parameter(
        "random_state",
        help="Random state for several application",
        default=42,
    )

    config_path = Parameter(
        "config_path",
        help="Config file path for training params",
        default=str(Path(__file__).parent / "conf" / "config.yml"),
    )

    input_file_path = Parameter(
        "input_file_path",
        help="Path to files containing input data",
        default=str(Path(__file__).parents[1] / "data"),
    )

    positive_interaction_threshold = Parameter(
        "positive_interaction_threshold",
        help="Threshold defining if an interactions question/answer is positive based on answer's score",
        default=1,
    )

    @step
    def start(self) -> None:
        "Check if input files are missing to trigger the download from google drive"
        import os
        from user_recommendation.data_preparation.data_loading.fetch_data import AVAILABLE_FILES

        self.all_exists = True
        for file in AVAILABLE_FILES:
            if not os.path.isfile(str(Path(self.input_file_path) / Path(file))):
                self.all_exists = False
                break

        self.next(self.download_data)

    @step
    def download_data(self) -> None:
        "Download data from google drive"
        from user_recommendation.data_preparation.data_loading.fetch_data import download_data

        if not self.all_exists:
            download_data()
        else:
            logger.info("Files already downloads, this step is skipped")
        self.next(self.read_data)

    @step
    def read_data(self) -> None:
        "Read users.json and questions.json as pandas dataframe"
        from user_recommendation.data_preparation.data_loading.fetch_data import read_data

        self.users = read_data("users.json", path=self.input_file_path)
        self.questions = read_data("questions.json", path=self.input_file_path)
        self.next(self.load_config)

    @step
    def load_config(self) -> None:
        """Load training config from yaml file"""
        import yaml

        with open(self.config_path, "r") as stream:
            config = yaml.load(stream, Loader=None)
        self.config = config.get("text_processing")
        logger.info(f"Config parsed: {self.config}")
        self.next(self.initialize_keywords_extractors_tfidf)

    @step
    def initialize_keywords_extractors_tfidf(self) -> None:
        """Initialize tfidf keyword extractor"""
        from user_recommendation.data_preparation.text_processing.keywords_extraction import (
            KeywordsExtractor,
            EKeywordExtractorTag,
        )

        nb_keywords = self.config.get("nb_keywords")
        self.extractor_tfidf_users = KeywordsExtractor(
            extraction_method=EKeywordExtractorTag.TFIDF,
            stop_words=self.config.get("stop_words"),
            strip_accents=self.config.get("strip_accents"),
            top_n=nb_keywords,
        )
        self.extractor_tfidf_questions = KeywordsExtractor(
            extraction_method=EKeywordExtractorTag.TFIDF,
            stop_words=self.config.get("stop_words"),
            strip_accents=self.config.get("strip_accents"),
            top_n=nb_keywords,
        )
        self.next(self.extract_users_keywords_tfidf, self.extract_questions_keywords_tfidf)

    @step
    def extract_users_keywords_tfidf(self) -> None:
        """Extract keywords from user textual data using tfidf"""
        nb_keywords = self.config.get("nb_keywords")
        self.tags_columns = [f"tags{i}" for i in range(1, nb_keywords + 1)]
        # A lot of about me are missing, replace it by empty string for processing
        self.users.about_me.fillna("", inplace=True)
        data_to_process = self.users.about_me.tolist()
        self.extractor_tfidf_users.fit(self.users.about_me.tolist())
        logger.info("Start extracting keywords from users")
        keywords = self._batch_tfidf(extractor=self.extractor_tfidf_users, data_to_batch=data_to_process)
        keywords_df = pd.DataFrame(keywords, columns=self.tags_columns)
        self.users_keywords_df = pd.concat([self.users, keywords_df], axis=1)
        logger.info(f"users_keywords_df shape: {self.users_keywords_df.shape}")
        self.next(self.join)

    @step
    def extract_questions_keywords_tfidf(self) -> None:
        """Extract keywords from question textual data using tfidf"""
        nb_keywords = self.config.get("nb_keywords")
        self.tags_columns = [f"tags{i}" for i in range(1, nb_keywords + 1)]
        data_to_process = self.questions.title.tolist()
        self.extractor_tfidf_questions.fit(self.questions.title.tolist())
        logger.info("Start extracting keywords from questions")
        keywords = self._batch_tfidf(extractor=self.extractor_tfidf_questions, data_to_batch=data_to_process)
        keywords_df = pd.DataFrame(keywords, columns=self.tags_columns)
        self.questions_keywords_df = pd.concat([self.questions, keywords_df], axis=1)
        logger.info(f"questions_keywords_df shape: {self.questions_keywords_df.shape}")
        self.next(self.join)

    @step
    def join(self, inputs) -> None:  # type: ignore
        """Merge data artifact"""
        self.tags_columns = inputs.extract_questions_keywords_tfidf.tags_columns
        self.extractor_tfidf_questions = inputs.extract_questions_keywords_tfidf.extractor_tfidf_questions
        self.extractor_tfidf_users = inputs.extract_users_keywords_tfidf.extractor_tfidf_users
        self.merge_artifacts(
            inputs,
            include=[
                "tags_columns",
                "extractor_tfidf_questions",
                "questions_keywords_df",
                "extractor_tfidf_users",
                "users_keywords_df",
            ],
        )
        self.next(self.end)

    @step
    def end(self) -> None:
        """End of flow"""
        pass
