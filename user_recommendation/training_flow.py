"""File where Training Flow is defined"""
from datetime import datetime, timezone
from pathlib import Path
from metaflow import FlowSpec, step, Parameter, card, current
from metaflow.cards import Image, Artifact, Markdown
import pandas as pd
from user_recommendation import logger


class TrainingModelFlow(FlowSpec):
    """Flow for training LightFM model"""

    @staticmethod
    def subsample_for_ranking(data_to_subsample: pd.DataFrame, random_state: int, frac: float) -> pd.DataFrame:
        """Sample some questions from test set cold start to predict

        Args:
            data_to_subsample: dataframe with questions unseen at training stage
            random_state: random state
            frac: rate of question to sample

        Returns:
            pd.DataFrame: question on which prediction will apply
        """
        questions_unique = data_to_subsample.drop_duplicates(subset="question_id")
        return questions_unique[questions_unique.data_type == "cold_start"].sample(
            frac=frac, random_state=random_state
        )[["question_id"]]

    random_state = Parameter(
        "random_state",
        help="Random state for several application",
        default=42,
    )
    is_tracked = Parameter(
        "is_tracked",
        help="Track model metrics for training stage",
        default=True,
    )
    show_plot = Parameter(
        "show_plot",
        help="Plot tracked metrics for training stage",
        default=True,
    )
    config_path = Parameter(
        "config_path",
        help="Config file path for training params",
        default=str(Path(__file__).parent / "conf" / "config.yml"),
    )

    @step
    def start(self) -> None:
        """Get all needed data from flow GenerateTrainTestValFlow latest successful run"""
        from metaflow import Flow, get_metadata

        logger.info(f"Using metadata provider: {get_metadata()}")

        run = Flow("GenerateTrainTestValFlow").latest_successful_run
        self.datasets = run.data.datasets
        logger.info("Get training, test, validation datasets")
        for fname, df in self.datasets:
            logger.info(f"{fname} shape: {df.shape}")
        self.answers = run["read_data"].task.data.answers
        logger.info(f"Get answers dataframe {self.answers.shape}")
        self.questions = run["read_data"].task.data.questions
        logger.info(f"Get questions dataframe {self.questions.shape}")
        self.users = run["read_data"].task.data.users
        logger.info(f"Get users dataframe {self.users.shape}")
        self.next(self.load_config)

    @step
    def load_config(self) -> None:
        """Load training config from yaml file"""
        import yaml

        with open(self.config_path, "r") as stream:
            config = yaml.load(stream, Loader=None)
        self.config = config.get("train_and_eval")
        logger.info(f"Config parsed: {self.config}")
        self.next(self.build_lightfm_dataset)

    @step
    def build_lightfm_dataset(self) -> None:
        """Build lightfm dataset from answers database"""
        from user_recommendation.training.lightfm_processing import get_lightfm_dataset

        self.lightfm_dataset = get_lightfm_dataset(self.answers)
        self.next(self.build_interactions)

    @step
    def build_interactions(self) -> None:
        """Build interactions for training, test and validation data"""
        from user_recommendation.training.lightfm_processing import get_interactions

        self.training_interactions_weigths = get_interactions(  # type: ignore
            data=self.datasets.training, dataset=self.lightfm_dataset, with_weights=False, obj_desc="training"
        )
        self.test_interactions_weigths = get_interactions(  # type: ignore
            data=self.datasets.test, dataset=self.lightfm_dataset, with_weights=False, obj_desc="test"
        )
        self.validation_interactions_weigths = get_interactions(  # type: ignore
            data=self.datasets.validation, dataset=self.lightfm_dataset, with_weights=False, obj_desc="validation"
        )
        self.next(self.train_model)

    @card(type='blank')
    @step
    def train_model(self) -> None:
        """Train Ligth FM model"""
        from user_recommendation.training.train import LigthFMTrainer

        trainer = LigthFMTrainer(dataset=self.lightfm_dataset)

        self.model_artifacts = trainer.fit(
            train_interactions=self.training_interactions_weigths[0],
            test_interactions=self.validation_interactions_weigths[0],
            epochs=self.config.get("epochs"),
            num_threads=self.config.get("num_threads"),
            is_tracked=self.is_tracked,
            show_plot=self.show_plot,
        )
        current.card.append(Markdown("# Training info  "))
        current.card.append(Markdown(f"Run id {current.run_id}  "))
        current.card.append(Markdown(f"Pathspec id {current.pathspec}  "))
        current.card.append(Markdown("## Config  "))
        current.card.append(Artifact(self.config))

        current.card.append(Markdown("## Model params  "))
        current.card.append(Artifact(self.model_artifacts[0].model.get_params()))

        if self.is_tracked:
            current.card.append(Markdown("## Tracked Metrics  "))
            current.card.append(Markdown("### Tracked Metrics values "))
            current.card.append(Artifact(self.model_artifacts[1]))
            if self.show_plot:
                current.card.append(Markdown("### Tracked Metrics Plot "))
                current.card.append(Image.from_matplotlib(self.model_artifacts[-1]))
        self.next(self.test_evaluation)

    @step
    def test_evaluation(self) -> None:
        """Evaluate test set"""
        from lightfm.evaluation import auc_score

        model = self.model_artifacts[0]
        logger.info(f"Evaluate test set")
        self.test_auc = model.evaluation_step(
            test_interactions=self.test_interactions_weigths[0],
            func=auc_score,
            train_interactions=self.training_interactions_weigths[0],
        )
        logger.info(f"Test auc: {self.test_auc}")
        self.next(self.test_prediction)

    @step
    def test_prediction(self) -> None:
        """Make ranking prediction on subsample of test set"""
        from user_recommendation.training.lightfm_processing import get_interactions

        model = self.model_artifacts[0]
        # Subsample questions to apply prediction
        subsample_questions = self.subsample_for_ranking(self.datasets.test, random_state=self.random_state, frac=0.05)
        logger.info(f"{subsample_questions.size} will be predict")
        # Iterate over all users will be too long for an example run. So we will randomly take a sample of users
        subsample_users_list = self.answers.drop_duplicates(subset="user_id").user_id.sample(n=3000).tolist()
        subsample_questions["user_id"] = [subsample_users_list for i in subsample_questions.index]
        subsample_questions = subsample_questions.explode("user_id").reset_index(drop=True)
        logger.info(f"Data to predict shape: {subsample_questions.shape}")

        logger.info("Get interaction for prediction")
        to_predict_interactions = get_interactions(  # type: ignore
            data=subsample_questions, dataset=self.lightfm_dataset, with_weights=False, obj_desc="data to predict"
        )[0]

        logger.info("Prediction is running")
        self.ranked_predictions = model.predict_rank(
            data=subsample_questions,
            interactions=to_predict_interactions,
            num_threads=self.config.get("num_threads"),
        )
        logger.info("Prediction Done")
        path = str(
            Path(__file__).parents[1]
            / "test_user_recommendation"
            / "integration_test"
            / f"{datetime.now(timezone.utc).isoformat()}.csv"
        )
        self.ranked_predictions.to_csv(
            path,
            index=False,
        )
        logger.info(f"Prediction saved at {path}")
        self.next(self.end)

    @step
    def end(self) -> None:
        pass
