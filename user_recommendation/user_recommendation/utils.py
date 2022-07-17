"""Some utils functions"""
from enum import Enum
import logging
import sys
import traceback
import pandas as pd
from typing import Optional, Type

from user_recommendation.errors import UserRecommendationException

# The format string to be used by the logger across projects.
FORMAT: str = "{asctime} :: {module}/{filename}/{funcName} :: {levelname} :: {message}"


def init_logger(name: Optional[str] = None) -> logging.Logger:
    """Initialize the logger for the application using parameters from environment."""
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()

    level: int = logging.DEBUG
    formatter: logging.Formatter = logging.Formatter(FORMAT, style="{")

    handler.setFormatter(formatter)
    handler.setStream(sys.stderr)

    logger.addHandler(handler)
    logger.setLevel(level)

    return logger


def log_raise(
    logger: logging.Logger,
    err: Exception,
    original_err: Optional[Exception] = None,
) -> None:
    """Log an error and raise exception

    Args:
        logger: logger instance
        err: custom exception thrown by cool job
        original_err: original exception custom exception wraps around, if any
            (default None)

    Raises:
        Exception: systematically
    """
    logger = logging.LoggerAdapter(logger, {"trace": traceback.format_exc()})
    logger.error(msg=str(err))
    if original_err:
        raise err from original_err
    raise err


def log_attribute_per_dataset(df_data: pd.DataFrame, attribute: str, logger: logging.Logger, desc: str) -> None:
    """Log metric with number of samples per category

    Args:
        df_data: dataframe that contains the data
        attribute: attribute on which group by
        logger: logger instance
        desc: metric description

    """
    nb_attr = {}
    temp = df_data.groupby(attribute).agg("count")
    for name, nb_samples in pd.DataFrame(temp.iloc[:, 0]).itertuples():
        nb_attr[str(name)] = nb_samples
    logger.info(f"{desc}: {nb_attr}")


def string_to_enum(
    enum: str, enum_class: Type[Enum], err_cls: Type[UserRecommendationException], logger: logging.Logger
) -> Enum:
    """Parse integer to an enumerator defined in job's API.

    Args:\
        enum: Input string to parse.
        enum_class: Enum class.
        err_cls: error to raise.
        logger: instance of logger.

    Returns:\
        Enum: return enum instance from input string.

    Raises:\
        Type[UserRecommendationException]: Raised if input string doesn't match enum instance.
    """
    try:
        enum_instance = enum_class(enum)
    except ValueError:
        err = err_cls(enum, enum_class.__doc__)  # type: ignore
        logger.error(str(err))
        raise err from ValueError
    return enum_instance
