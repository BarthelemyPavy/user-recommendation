"""Some utils functions"""

# The format string to be used by the logger across projects.
import logging
import sys
import traceback
import pandas as pd
from typing import Optional


FORMAT: str = "{asctime} :: {funcName} :: {levelname} :: {message}"


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
    logger.error(msg=str(err), trace=traceback.format_exc())  # type: ignore
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
    logger.info(desc, **nb_attr)
