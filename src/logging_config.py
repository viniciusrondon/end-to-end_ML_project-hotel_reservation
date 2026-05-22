"""Centralized logging for pipeline stages."""

import inspect
import logging
import os
from datetime import datetime
from pathlib import Path

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    When name is __main__ (script run directly), infers the actual script
    name from the caller's __file__ so logs show e.g. 'test_logger' instead.
    """
    if name == "__main__":
        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            caller_file = frame.f_back.f_globals.get("__file__")
            if caller_file:
                name = Path(caller_file).stem
            else:
                name = "root"
        else:
            name = "root"
    return logging.getLogger(name)


"""tester_logger

from src.logging_config import get_logger


logger = get_logger(__name__)


logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")"""
