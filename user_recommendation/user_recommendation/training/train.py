"""Train models"""
import copy
import numpy as np
import pandas as pd
import numpy.typing as npt
import seaborn as sns
from typing import Callable, Optional, Tuple, TypeVar
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
from tqdm import tqdm

from user_recommendation.training.lightfm_processing import COOMatrix
from user_recommendation.utils import log_raise
from user_recommendation.errors import BadEvaluationMetric
from user_recommendation import logger


Plot = TypeVar("Plot")
EvaluationFuncType = Callable[[LightFM, COOMatrix, COOMatrix], npt.NDArray[np.float_]]


class LigthFMTrainer:
    """Trainer for light fm models

    Attributes:
        _model: Light FM model
        _question_id_map:
        _question_feature_map:
        _user_id_map:
        _user_feature_map:
    """

    _AVAILABLE_VALIDATION: dict[str, EvaluationFuncType] = {  # type: ignore
        "precision": precision_at_k,
        "recall": recall_at_k,
        "auc": auc_score,
    }
    _model: LightFM
    _question_id_map: dict[str, int]
    _question_feature_map: dict[str, int]
    _user_id_map: dict[str, int]
    _user_feature_map: dict[str, int]

    def __init__(self, dataset: Dataset, **kwargs: int) -> None:
        """Class Constructor

        Args:
            dataset: light fm dataset fitted at previsou step
        """
        self._question_id_map = dataset.mapping()[0]
        self._question_feature_map = dataset.mapping()[1]
        self._user_id_map = dataset.mapping()[2]
        self._user_feature_map = dataset.mapping()[3]
        self._model = LightFM(**kwargs)

    @property
    def model(self) -> LightFM:
        """Get light fm model"""
        return self._model

    def fit(
        self,
        train_interactions: COOMatrix,
        test_interactions: Optional[COOMatrix] = None,
        epochs: int = 1,
        num_threads: int = 1,
        is_tracked: bool = True,
        validation_metrics: Optional[list[str]] = None,
        reset_state: bool = False,
        **kwargs: int,
    ) -> Tuple[LightFM, Optional[pd.DataFrame]]:
        """Fit ligth fm model

        Args:
            train_interactions: Matrix containing user-item interactions for train
            test_interactions: Matrix containing user-item interactions for validation. Defaults to None.
            epochs: number of epoch. Defaults to 1.
            num_threads: number of threads to use. Defaults to 1.
            is_tracked: if model metrics need to be track over epoch.\n
                         If True, for each epoch an evaluation with train and validation interactions will be done\n
                         and save then plot.
                        Defaults to True.
            show_plot: Should we plot metrics or not. Defaults to True.
            reset_state: If True, reset all parameters to fit from scratch

        Returns:
            Tuple[LightFM, Optional[pd.DataFrame]]: A tuple containing the fitted model
                            and the monitoring of tracked metris if 'is_tracked=True'

        """
        if not is_tracked:
            user_features = kwargs.get("user_features", None)
            kwargs.pop("user_features", None)
            item_features = kwargs.get("item_features", None)
            kwargs.pop("item_features", None)
            self._model.fit(
                interactions=train_interactions,
                epochs=epochs,
                num_threads=num_threads,
                verbose=True,
                user_features=user_features,
                item_features=item_features,
                **kwargs,
            )
            to_return = (
                self,
                None,
            )
        else:
            if test_interactions == None:
                log_raise(logger=logger, err=ValueError("If is_tracked=True, test_interactions should not be None."))

            if reset_state:
                self._model._reset_state()

            to_return = self._track_model(
                train_interactions=train_interactions,
                test_interactions=test_interactions,
                validation_metrics=validation_metrics,
                epochs=epochs,
                num_threads=num_threads,
                **kwargs,
            )
        return to_return

    def _track_model(
        self,
        train_interactions: COOMatrix,
        test_interactions: COOMatrix,
        validation_metrics: Optional[list[str]] = None,
        epochs: int = 1,
        num_threads: int = 1,
        **kwargs: int,
    ) -> Tuple[LightFM, pd.DataFrame]:
        """Function to record model's performance at each epoch, formats the performance into tidy format,
        plots the performance and outputs the performance data.
        Args:
            train_interactions: train interactions set
            test_interactions: test interaction set
            epochs: Number of epochs to run, optional
            num_threads: Number of parallel threads to use, optional
            **kwargs: other keyword arguments to be passed down
        Returns:
            LightFM model, pandas.DataFrame, matplotlib axes:
            - Best Fitted model based on test auc
            - Performance traces of the fitted model
            - Side effect of the method
        """
        if not validation_metrics:
            validation_metrics = ["auc"]
        else:
            # Because auc it's used to keep the best model during training stage
            validation_metrics = validation_metrics + ["auc"] if "auc" not in validation_metrics else validation_metrics
        # initialising temp data storage
        model_track = pd.DataFrame(columns=["epoch", "stage", "metric", "value"])
        # default value for best model selection
        best_model, best_auc_eval = None, -1

        # fit model and store train/test metrics at each epoch
        loop_epoch = tqdm(range(epochs))
        loop_epoch.set_description(f"Training Epoch")
        for epoch in loop_epoch:
            self._model.fit_partial(interactions=train_interactions, epochs=1, num_threads=num_threads, **kwargs)

            model_track = self._loop_over_evaluation_metrics(
                metric_storage=model_track,
                train_interactions=train_interactions,
                test_interactions=test_interactions,
                validation_metrics=validation_metrics,
                epoch=epoch,
                **kwargs,
            )

            last_auc_value = (
                model_track[(model_track.stage == "validation") & (model_track.metric == "auc")].value.tail(1).item()
            )
            if last_auc_value > best_auc_eval:
                best_model, best_auc_eval = self._model, last_auc_value

        self._model = best_model
        # replace the metric keys to improve visualisation
        metric_keys = {"precision": "Precision", "recall": "Recall", "auc": "ROC AUC"}
        model_track.metric.replace(metric_keys, inplace=True)

        return self, model_track

    def _loop_over_evaluation_metrics(
        self,
        metric_storage: pd.DataFrame,
        train_interactions: COOMatrix,
        test_interactions: COOMatrix,
        validation_metrics: list[str],
        epoch: int,
        **kwargs: int,
    ) -> pd.DataFrame:
        """Loop over evaluation metrics in order to evaluate model on train and validation test for each epoch

        Args:
            metric_storage: Storage to save evaluation scores
            validation_metrics: List of metrics

        Returns:
            pd.DataFrame: All evaluation scores
        """
        loop_metrics = tqdm(validation_metrics)
        loop_metrics.set_description(f"Iterate over evaluation metrics")
        kwargs.pop("sample_weight", None)
        for metric in loop_metrics:
            logger.info(f"Evaluate training at epoch: {str(epoch)} with metric: {metric}")
            eval_kwargs = copy.deepcopy(kwargs)
            try:
                if metric == "auc":
                    eval_kwargs.pop("k", None)
                train_metric = self.evaluation_step(
                    test_interactions=train_interactions, func=self._AVAILABLE_VALIDATION.get(metric), **eval_kwargs  # type: ignore
                )
                metric_storage = metric_storage.append(
                    {"epoch": epoch, "stage": "train", "metric": metric, "value": train_metric}, ignore_index=True
                )
                test_metric = self.evaluation_step(
                    test_interactions=test_interactions,
                    func=self._AVAILABLE_VALIDATION.get(metric),  # type: ignore
                    train_interactions=train_interactions,
                    **eval_kwargs,
                )
                metric_storage = metric_storage.append(
                    {"epoch": epoch, "stage": "validation", "metric": metric, "value": test_metric}, ignore_index=True
                )
            except KeyError as err:
                log_raise(
                    logger=logger,
                    err=BadEvaluationMetric(metric=metric, available_metrics=list(self._AVAILABLE_VALIDATION.keys())),
                    original_err=err,
                )
        return metric_storage

    def evaluation_step(
        self,
        test_interactions: COOMatrix,
        func: EvaluationFuncType,  # type: ignore
        train_interactions: Optional[COOMatrix] = None,
        **kwargs: int,
    ) -> float:
        """Evaluate model

        Args:
            test_interactions: see fit docstring
            train_interactions: see fit docstring
            func: evaluation function to call

        Returns:
            float: metric compute by evaluation func
        """
        val_metric: float = func(self._model, test_interactions, train_interactions, **kwargs).mean()
        return val_metric

    @staticmethod
    def model_perf_plots(df: pd.DataFrame) -> Plot:
        """Function to plot model performance metrics.

        Args:
            df: Dataframe in tidy format, with ['epoch','level','value'] columns

        Returns:
            object: matplotlib axes
        """
        g = sns.FacetGrid(df, col="metric", hue="stage", col_wrap=2, sharey=False)
        g = g.map(sns.scatterplot, "epoch", "value").add_legend()
        fig: Plot = g.figure
        return fig

    def predict_rank(
        self,
        data: pd.DataFrame,
        interactions: COOMatrix,
        num_threads: int,
        question_features: Optional[COOMatrix] = None,
        user_features: Optional[COOMatrix] = None,
        max_rank: int = 20,
    ) -> pd.DataFrame:
        """For each question give a list of user_id suscpetible to answer ranked by prediction score

        Args:
            data: dataframe of all users, items and ratings as loaded
            interactions: user-item interaction
            num_threads: number of parallel computation threads
            question_features: User weights over features
            user_features: Item weights over features
            max_rank: Maximum of user recommended

        Returns:
            pd.DataFrame: data containing list of user_id suscpetible to answer ranked by prediction score
        """

        df_pred = self.predict(
            data=data,
            interactions=interactions,
            num_threads=num_threads,
            question_features=question_features,
            user_features=user_features,
        )
        return (
            df_pred.groupby("question_id")
            .apply(lambda x: x.sort_values(by="prediction", ascending=False).head(max_rank).user_id.tolist())
            .reset_index()
        )

    def predict(
        self,
        data: pd.DataFrame,
        interactions: COOMatrix,
        num_threads: int,
        question_features: Optional[COOMatrix] = None,
        user_features: Optional[COOMatrix] = None,
    ) -> pd.DataFrame:
        """Predict one or several user-item score and map ids

        Args:
            data: dataframe of all users, items and ratings as loaded
            interactions: user-item interaction
            num_threads: number of parallel computation threads
            question_features: question weights over features
            user_features: user weights over features

        Returns:
            pd.DataFrame: data containing predictions score
        """
        tqdm.pandas()
        questions, users = [], []
        user = list(data.user_id.unique())
        for question in data.question_id.unique():
            question = [question] * len(user)
            questions.extend(question)
            users.extend(user)
        all_predictions = pd.DataFrame(data={"question_id": questions, "user_id": users})
        all_predictions["uid"] = all_predictions.question_id.map(self._question_id_map)
        all_predictions["iid"] = all_predictions.user_id.map(self._user_id_map)

        dok_weights = interactions.todok()  # type: ignore
        all_predictions["rating"] = all_predictions.apply(lambda x: dok_weights[x.uid, x.iid], axis=1)

        all_predictions["prediction"] = all_predictions.progress_apply(
            lambda x: self._model.predict(
                user_ids=np.array([x["uid"]], dtype=np.int32),
                item_ids=np.array([x["iid"]], dtype=np.int32),
                user_features=question_features,
                item_features=user_features,
                num_threads=num_threads,
            )[0],
            axis=1,
        )

        return all_predictions[["question_id", "user_id", "prediction"]]
