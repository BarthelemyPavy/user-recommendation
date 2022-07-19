"""File where Training Flow is defined"""
from datetime import datetime, timezone
from pathlib import Path
from metaflow import FlowSpec, step, Parameter, card, current, metadata
from metaflow.cards import Image, Artifact, Markdown
import pandas as pd
from user_recommendation import logger


metadata("local@" + str(Path(__file__).parents[1]))


class TrainingModelFlow(FlowSpec):
    """Flow for training LightFM model.\n
    In this flow we will:\n
        - Load data artifact from both GenerateTrainTestValFlow and TextProcessingFlow.
        - Build lightfm dataset and create interactions.
        - Build users and questions features for Content Based part of recommendation.
        - Train the model as pure Collaborative filtering model and Hybrid model.
        - Test models on test set.
        - Select best model based on test results.
        - Make a prediction of a subsample of test result as: question_id, [user_1, user_2, ...]
        - Save prediction results as csv
        - Generate report with all metrics from training
    """

    @staticmethod
    def _subsample_for_ranking(data_to_subsample: pd.DataFrame, random_state: int, frac: float) -> pd.DataFrame:
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
    is_tracked = Parameter("is_tracked", help="Track model metrics for training stage", default=False, type=bool)
    config_path = Parameter(
        "config_path",
        help="Config file path for training params",
        default=str(Path(__file__).parent / "conf" / "config.yml"),
    )
    reduce_train_set = Parameter(
        "reduce_train_set",
        help="expecting a float 0<x<=1 defining the rate of training set to keep. This could be use for reduce the training time.",
        default=0,
        type=float,
    )

    @step
    def start(self) -> None:
        """Get all needed data from flow GenerateTrainTestValFlow latest successful run"""
        from metaflow import Flow, get_metadata

        logger.info(f"Using metadata provider: {get_metadata()}")
        # Get data artifacts from GenerateTrainTestValFlow

        run_test_train_split = Flow("GenerateTrainTestValFlow").latest_successful_run
        self.datasets = run_test_train_split.data.datasets
        logger.info("Get training, test, validation datasets")
        for fname, df in self.datasets:
            logger.info(f"{fname} shape: {df.shape}")
        self.answers = run_test_train_split["read_data"].task.data.answers
        logger.info(f"Get answers dataframe {self.answers.shape}")
        self.questions = run_test_train_split["read_data"].task.data.questions
        logger.info(f"Get questions dataframe {self.questions.shape}")
        self.users = run_test_train_split["read_data"].task.data.users
        logger.info(f"Get users dataframe {self.users.shape}")

        # Get data artifacts from TextProcessingFlow
        run_text = Flow("TextProcessingFlow").latest_successful_run
        self.questions_keywords_df = run_text["join"].task.data.questions_keywords_df
        logger.info(f"Get questions_keywords_df dataframe {self.questions_keywords_df.shape}")
        # Fill nan value by "", as we've seen in data exploration step
        # only textual data was missing for users
        self.users_keywords_df = run_text["join"].task.data.users_keywords_df.fillna("")
        logger.info(f"Get users_keywords_df dataframe {self.users_keywords_df.shape}")
        self.tags_columns = run_text["join"].task.data.tags_columns
        logger.info(f"Number of tags per question/user {len(self.tags_columns)}")

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

        self.lightfm_dataset = get_lightfm_dataset(
            users=self.users,
            questions=self.questions,
            user_features=self.users_keywords_df,
            question_features=self.questions_keywords_df,
            tags_columns=self.tags_columns,
        )
        self.next(self.build_interactions)

    @step
    def build_interactions(self) -> None:
        """Build interactions for training, test and validation data"""
        from user_recommendation.training.lightfm_processing import get_interactions
        from sklearn.model_selection import train_test_split

        if self.reduce_train_set > 0:
            logger.warning(
                f"""A reduce_train_set value is passed (={self.reduce_train_set}).
                           This means that the train dataset will be reduced by {(1-self.reduce_train_set)*100}%"""
            )
            self.datasets.training, _ = train_test_split(
                self.datasets.training,
                test_size=(1 - self.reduce_train_set),
                random_state=self.random_state,
                stratify=self.datasets.training.question_label,
            )
            logger.info(f"New Training dataset shape: {self.datasets.training.shape}")

        self.training_interactions_weigths = get_interactions(  # type: ignore
            data=self.datasets.training, dataset=self.lightfm_dataset, with_weights=True, obj_desc="training"
        )
        matrix = self.training_interactions_weigths[0].copy()
        sparsity = 1.0 - (matrix.count_nonzero() / float(matrix.toarray().size))
        logger.info(f"Sparsity of the training matrix: {sparsity}")
        self.test_interactions_weigths = get_interactions(  # type: ignore
            data=self.datasets.test, dataset=self.lightfm_dataset, with_weights=True, obj_desc="test"
        )
        self.validation_interactions_weigths = get_interactions(  # type: ignore
            data=self.datasets.validation, dataset=self.lightfm_dataset, with_weights=True, obj_desc="validation"
        )
        self.next(self.build_users_features)

    @step
    def build_users_features(self) -> None:
        """Build users features matrix"""
        from user_recommendation.training.lightfm_processing import get_users_features

        self.users_features = get_users_features(  # type: ignore
            data=self.users_keywords_df, dataset=self.lightfm_dataset, tags_column=self.tags_columns
        )
        self.next(self.build_questions_features)

    @step
    def build_questions_features(self) -> None:
        """Build users features matrix"""
        from user_recommendation.training.lightfm_processing import get_questions_features

        self.questions_features = get_questions_features(  # type: ignore
            data=self.questions_keywords_df, dataset=self.lightfm_dataset, tags_column=self.tags_columns
        )
        self.next(self.initialize_lightfm_trainer)

    @step
    def initialize_lightfm_trainer(self) -> None:
        """Initialize models for training"""
        from user_recommendation.training.train import LigthFMTrainer

        self.trainer_cf = LigthFMTrainer(
            dataset=self.lightfm_dataset,
            no_components=self.config.get("no_components"),
            random_state=self.random_state,
            loss=self.config.get("loss"),
            learning_rate=self.config.get("learning_rate"),
            learning_schedule=self.config.get("learning_schedule"),
        )
        self.trainer_hybrid = LigthFMTrainer(
            dataset=self.lightfm_dataset,
            no_components=self.config.get("no_components"),
            random_state=self.random_state,
            loss=self.config.get("loss"),
            learning_rate=self.config.get("learning_rate"),
            learning_schedule=self.config.get("learning_schedule"),
        )
        self.next(self.train_model_hybrid)

    @step
    def train_model_hybrid(self) -> None:
        """Train Ligth FM model"""

        self.model_artifacts_hybrid = self.trainer_hybrid.fit(
            train_interactions=self.training_interactions_weigths[0],
            test_interactions=self.validation_interactions_weigths[0],
            epochs=self.config.get("epochs"),
            num_threads=self.config.get("num_threads"),
            is_tracked=self.is_tracked,
            user_features=self.questions_features,
            item_features=self.users_features,
            sample_weight=self.training_interactions_weigths[1],
        )
        self.next(self.train_model_cf)

    @step
    def train_model_cf(self) -> None:
        """Train Ligth FM model"""

        self.model_artifacts_cf = self.trainer_cf.fit(
            train_interactions=self.training_interactions_weigths[0],
            test_interactions=self.validation_interactions_weigths[0],
            epochs=self.config.get("epochs"),
            num_threads=self.config.get("num_threads"),
            is_tracked=self.is_tracked,
            sample_weight=self.training_interactions_weigths[1],
        )
        self.next(self.test_evaluation)

    @step
    def test_evaluation(self) -> None:
        """Evaluate test set"""
        from lightfm.evaluation import auc_score

        # Evaluate CF model
        model_cf = self.model_artifacts_cf[0]
        logger.info(f"Evaluate test set for CF model")

        self.test_auc_cf = model_cf.evaluation_step(
            test_interactions=self.test_interactions_weigths[0],
            func=auc_score,
            train_interactions=self.training_interactions_weigths[0],
            num_threads=self.config.get("num_threads"),
        )
        logger.info(f"Test Model CF auc: {self.test_auc_cf}")
        # Evaluate Hybrid model
        model_hybrid = self.model_artifacts_hybrid[0]
        logger.info(f"Evaluate test set for Hybrid model")

        self.test_auc_hybrid = model_hybrid.evaluation_step(
            test_interactions=self.test_interactions_weigths[0],
            func=auc_score,
            train_interactions=self.training_interactions_weigths[0],
            user_features=self.questions_features,
            item_features=self.users_features,
            num_threads=self.config.get("num_threads"),
        )
        logger.info(f"Test Model Hybrid auc: {self.test_auc_hybrid}")
        self.next(self.select_best_model)

    @step
    def select_best_model(self) -> None:
        """Select best model between Pure CF and Hybrid"""
        if self.test_auc_cf >= self.test_auc_hybrid:
            self.best_model = self.model_artifacts_cf[0]
            logger.info("And the winner is: Collaborative Filtering Model!!!")
            self.best_model_name = "cf"
        else:
            self.best_model = self.model_artifacts_hybrid[0]
            logger.info("And the winner is: Hybrid Model!!!")
            self.best_model_name = "hybrid"
        self.next(self.test_prediction)

    @step
    def test_prediction(self) -> None:
        """Make ranking prediction on subsample of test set"""
        from user_recommendation.training.lightfm_processing import get_interactions

        # Subsample questions to apply prediction
        subsample_questions = self._subsample_for_ranking(self.datasets.test, random_state=self.random_state, frac=0.01)
        logger.info(f"{subsample_questions.size} questions will be predict")
        # Iterate over all users will be too long for an example run. So we will randomly take a sample of users
        subsample_users_list = self.answers.drop_duplicates(subset="user_id").user_id.sample(n=20000).tolist()
        subsample_questions["user_id"] = [subsample_users_list for i in subsample_questions.index]
        subsample_questions = subsample_questions.explode("user_id").reset_index(drop=True)
        logger.info(f"Data to predict shape: {subsample_questions.shape}")

        logger.info("Get interaction for prediction")
        to_predict_interactions = get_interactions(  # type: ignore
            data=subsample_questions, dataset=self.lightfm_dataset, with_weights=False, obj_desc="data to predict"
        )[0]

        logger.info("Prediction is running")
        if self.best_model_name == "hybrid":
            self.ranked_predictions = self.best_model.predict_rank(
                data=subsample_questions,
                interactions=to_predict_interactions,
                num_threads=self.config.get("num_threads"),
                question_features=self.questions_features,
                user_features=self.users_features,
            )
        elif self.best_model_name == "cf":
            self.ranked_predictions = self.best_model.predict_rank(
                data=subsample_questions,
                interactions=to_predict_interactions,
                num_threads=self.config.get("num_threads"),
            )
        logger.info("Prediction Done")
        self.next(self.save_predictions)

    @step
    def save_predictions(self) -> None:
        """Save predictions as csv into user_recommendation/test_user_recommendation/integration_test/"""
        path = str(
            Path(__file__).parent
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

    @card
    @step
    def end(self) -> None:
        """Generate Training report\n

        Run:
            > python user_recommendation/training_flow.py card view end

        At the end of the run to see the report
        """

        current.card.append(Markdown("# Models Report  "))
        current.card.append(Markdown("## Test Set Results  "))
        current.card.append(Markdown("Collaborative Filtering  "))
        current.card.append(Artifact(self.test_auc_cf))
        current.card.append(Markdown("Hybrid  "))
        current.card.append(Artifact(self.test_auc_hybrid))

        # CF model
        current.card.append(Markdown("## Collaborative Filtering Model  "))
        current.card.append(Markdown("### Model params  "))
        current.card.append(Artifact(self.model_artifacts_cf[0].model.get_params()))

        if self.is_tracked:
            current.card.append(Markdown("### Tracked Metrics "))
            current.card.append(Artifact(self.model_artifacts_cf[1]))
            if self.show_plot:
                current.card.append(Markdown("### Tracked Metrics Plot "))
                current.card.append(
                    Image.from_matplotlib(self.model_artifacts_cf[0].model_perf_plots(self.model_artifacts_cf[1]))
                )

        # Hybrid model
        current.card.append(Markdown("## Hybrid Model  "))
        current.card.append(Markdown("### Model params  "))
        current.card.append(Artifact(self.model_artifacts_hybrid[0].model.get_params()))

        if self.is_tracked:
            current.card.append(Markdown("### Tracked Metrics "))
            current.card.append(Artifact(self.model_artifacts_hybrid[1]))
            if self.show_plot:
                current.card.append(Markdown("### Tracked Metrics Plot "))
                current.card.append(
                    Image.from_matplotlib(
                        self.model_artifacts_hybrid[0].model_perf_plots(self.model_artifacts_hybrid[1])
                    )
                )


if __name__ == "__main__":
    TrainingModelFlow()
