"""Entry point file for flow generation"""
import fire
from enum import Enum
from pathlib import Path
from metaflow import metadata
from train_test_split_flow import GenerateTrainTestValFlow
from training_flow import TrainingModelFlow
from text_processing_flow import TextProcessingFlow
from user_recommendation import logger
from user_recommendation.utils import string_to_enum
from user_recommendation.errors import InvalidTag

metadata("local@" + str(Path(__file__).parents[1]))


class EFlowTags(Enum):
    """Tags that can be used to run different flow

    Attributes:\
        GENERATE_EMBEDDINGS: execute nodes that generate embeddings from input features
        SPLIT_DATASET: split data into train test val datasets
        TRAINING: train model
        ALL: run all flows

    """

    TEXT_PROCESSING = "text_processing"
    SPLIT_DATASET = "split_dataset"
    TRAINING = "training"
    ALL = "all"


def flow_trigger(**kwargs: dict[str, str]) -> None:
    """Use Metaflow tag argument to choose which flow to trigger.
    It's a workaround and not recommended."""
    tag: str = kwargs.get("tag")  # type: ignore
    flow_tag = string_to_enum(tag, EFlowTags, InvalidTag, logger)
    if flow_tag == EFlowTags.SPLIT_DATASET:
        GenerateTrainTestValFlow()
    if flow_tag == EFlowTags.TEXT_PROCESSING:
        TextProcessingFlow()
    if flow_tag == EFlowTags.TRAINING:
        TrainingModelFlow()
    if flow_tag == EFlowTags.ALL:
        GenerateTrainTestValFlow()
        TrainingModelFlow()


if __name__ == "__main__":
    fire.Fire(flow_trigger)
