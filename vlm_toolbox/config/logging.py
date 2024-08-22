import logging
import os
import sys
from typing import Union

from config.path import EXPERIMENTS_LOGGING_DIR
from loguru import logger

def init_logging(level: Union[int, str] = logging.INFO) -> None:
    """
    Initialize logging configuration using loguru.

    Args:
        level (Union[int, str]): The logging level. Default is logging.INFO.
    """
    logger.remove()

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    logging_file_path = os.path.join(EXPERIMENTS_LOGGING_DIR, "logs.log")

    logger.configure(
        handlers=[
            {"sink": sys.stdout, "level": level, "colorize": True},
            {"sink": logging_file_path, "level": level, "enqueue": True},
        ]
    )

def create_logger(**binding_params: dict) -> logging.Logger:
    """
    Retrieve a logger instance with optional bound parameters.

    Args:
        **binding_params (dict): Optional parameters to bind to the logger.

    Returns:
        logger: A loguru logger instance.
    """
    if not binding_params:
        return logger
    return logger.bind(**binding_params)

__all__ = ["init_logging", "get_logger"]
