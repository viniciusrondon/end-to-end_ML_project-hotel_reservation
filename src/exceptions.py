"""Custom domain-specific exceptions."""

import sys
from typing import Any


class ProjectError(Exception):
    """Base exception for project-specific failures."""

    def __init__(self, error_message: str, error_details: Any = None) -> None:
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_details)

    @staticmethod
    def get_detailed_error_message(error_message: str, error_details: Any = None) -> str:
        _exc_type, exc_val, exc_tb = sys.exc_info()
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            cause = f" | cause: {exc_val}" if exc_val else ""
            return (
                f"Error occurred in file: {file_name} at line number: {line_number} "
                f"error message: {error_message}{cause}"
            )
        if isinstance(error_details, BaseException):
            return f"{error_message} | cause: {error_details}"
        return error_message
    
    def __str__(self):
        return self.error_message


class MissingColumnsError(ProjectError):
    """Raised when required columns are missing from a dataset."""

    pass


class SchemaDriftError(ProjectError):
    """Raised when data schema differs from expected."""

    pass


class CorruptedArtifactError(ProjectError):
    """Raised when an artifact file is corrupted or invalid."""

    pass


class InvalidConfigError(ProjectError):
    """Raised when configuration is invalid or inconsistent."""

    pass


"""tester_exception.py -> create to test the exceptions
from src.exceptions import ProjectError
from src.logging_config import get_logger
import sys

logger = get_logger(__name__)


def divide_numbers(a: int, b: int) -> float:
    try:
        result = a / b
        logger.info(f"Result of division: {result}")
        return result
    except Exception as e:
        logger.error("Exception occurred while dividing numbers")
        raise ProjectError("Exception occurred while dividing numbers", sys)

if __name__ == "__main__":
    try:
        logger.info("Starting the program")
        divide_numbers(10, 0)
    except ProjectError as ce:
        logger.error(str(ce))"""
