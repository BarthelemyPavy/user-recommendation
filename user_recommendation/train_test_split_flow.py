"""File where train test split Flow is defined"""
from pathlib import Path
from metaflow import FlowSpec, step, Parameter, metadata
from user_recommendation import logger


metadata("local@" + str(Path(__file__).parents[1]))


class GenerateTrainTestValFlow(FlowSpec):
    """Flow used to generate train test and validation data.\n
    In this flow we will:\n
        - Download input data.
        - Preprocess data to split.
        - Apply a cold start split.
        - Apply a warm start split.
        - Wrap train, validation and test sets into an object.
    """

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
        "Download data from google drive and store it to data folder"
        from user_recommendation.data_preparation.data_loading.fetch_data import download_data

        if not self.all_exists:
            download_data()
        else:
            logger.info("Files already downloads, this step is skipped")
        self.next(self.read_data)

    @step
    def read_data(self) -> None:
        "Read users.json, answers.json and questions.json as pandas dataframe"
        from user_recommendation.data_preparation.data_loading.fetch_data import read_data

        self.users = read_data("users.json", path=self.input_file_path)
        self.answers = read_data("answers.json", path=self.input_file_path)
        self.questions = read_data("questions.json", path=self.input_file_path)
        self.next(self.load_config)

    @step
    def load_config(self) -> None:
        """Load training config from yaml file.
        This file contains parameters for train test split generation"""
        import yaml

        with open(self.config_path, "r") as stream:
            config = yaml.load(stream, Loader=None)
        self.config = config.get("data_split")
        logger.info(f"Config parsed: {self.config}")
        self.next(self.preprocess_dataframes)

    @step
    def preprocess_dataframes(self) -> None:
        """Preprocess input dataframes to create ready to use dataframe for train test validation split"""
        from user_recommendation.data_preparation.split_datasets.train_test_split import PreprocessDataframes

        preprocesser = PreprocessDataframes(questions=self.questions, answers=self.answers, users=self.users)
        self.preprocess_df = preprocesser.execute(positive_interactions_threshold=self.positive_interaction_threshold)
        self.next(self.cold_start_split)

    @step
    def cold_start_split(self) -> None:
        """Split data into cold start test and validation"""
        from user_recommendation.data_preparation.split_datasets.train_test_split import ColdStartSplit

        spliter = ColdStartSplit(preprocessed_df=self.preprocess_df)
        self.test_df_cs, self.val_df_cs = spliter.execute(
            test_val_size=self.config.get("test_val_size"),
            split_size=self.config.get("split_cold_warm_size"),
            random_state=self.random_state,
        )
        self.next(self.warm_start_split)

    @step
    def warm_start_split(self) -> None:
        """Split data into warm start test and validation"""
        from user_recommendation.data_preparation.split_datasets.train_test_split import WarmStartSplit

        cold_start_ids = list(set(self.test_df_cs.question_id.tolist())) + list(
            set(self.val_df_cs.question_id.tolist())
        )
        spliter = WarmStartSplit(preprocessed_df=self.preprocess_df, cold_start_ids=cold_start_ids)
        self.test_df_ws, self.val_df_ws = spliter.execute(
            split_size=self.config.get("split_cold_warm_size"),
            random_state=self.random_state,
        )
        self.next(self.end)

    @step
    def end(self) -> None:
        """Build training data and pack training, test and validation dataframes into datasets object"""
        from user_recommendation.data_preparation.split_datasets.train_test_split import CreateDatasetsObject

        creator = CreateDatasetsObject(preprocessed_df=self.preprocess_df)
        self.datasets = creator.execute(
            test_cs=self.test_df_cs, test_ws=self.test_df_ws, val_cs=self.val_df_cs, val_ws=self.val_df_ws
        )
        pass


if __name__ == "__main__":
    GenerateTrainTestValFlow()
