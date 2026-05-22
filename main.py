"""
Project structure automation for hotel reservation ML.

Creates the repository layout (directories and initial files) safely,
without overwriting or deleting any existing resources.
"""

from pathlib import Path
from typing import Final


# ---------------------------------------------------------------------------
# Structure definition: (path, content or None for dirs)
# None = directory only; str = file path with optional content
# ---------------------------------------------------------------------------

DIRS: Final[list[Path]] = [
    Path("src"),
    Path("artifacts"),
    Path("pipelines"),
    Path("notebooks"),
    Path("config"),
    Path("utils"),
    Path("templates"),
    Path("static"),
    Path("static/css"),
    Path("static/images"),
    Path("static/js"),
]

FILES: Final[dict[str, str]] = {
    "src/__init__.py": "",
    "src/logging_config.py": '''"""Centralized logging for pipeline stages."""

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

"""tester_logger.py -> create to test the logging config

from src.logging_config import get_logger


logger = get_logger(__name__)


logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")"""
''',
    "src/exceptions.py": '''"""Custom domain-specific exceptions."""

import traceback
import sys
from typing import Any


class ProjectError(Exception):
    """Base exception for project-specific failures."""

    def __init__(self, error_message: str, error_details: sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_details)

    @staticmethod
    def get_detailed_error_message(error_message: str, error_details: sys):
        _ , _ , exc_tb = error_details.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = f"Error occurred in file: {file_name} at line number: {line_number} error message: {error_message}"

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
''',
    "artifacts/.gitkeep": "",
    "pipelines/__init__.py": "",
    "notebooks/.gitkeep": "",
    "config/__init__.py": "",
    "config/.gitkeep": "",
    "utils/__init__.py": "",
    "templates/.gitkeep": "",
    "static/.gitkeep": "",
    "static/css/.gitkeep": "",
    "static/images/.gitkeep": "",
    "static/js/.gitkeep": "",
}


def _ensure_dir(path: Path, base: Path) -> bool:
    """Create directory if it does not exist. Returns True if created."""
    full = base / path
    if full.exists():
        if full.is_dir():
            return False
        raise FileExistsError(
            f"Cannot create directory {path}: path exists as file"
        )
    full.mkdir(parents=True, exist_ok=True)
    return True


def _ensure_file(rel_path: str, content: str, base: Path) -> bool:
    """Create file if it does not exist. Never overwrites. Returns True if created."""
    full = base / rel_path
    if full.exists():
        if full.is_file():
            return False
        raise FileExistsError(
            f"Cannot create file {rel_path}: path exists as directory"
        )
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    return True


def scaffold(base_dir: Path | None = None) -> dict[str, list[str]]:
    """
    Create project structure without overwriting existing resources.

    Args:
        base_dir: Root directory for scaffolding. Defaults to script directory.

    Returns:
        Dict with keys 'created_dirs' and 'created_files' listing created paths.
    """
    base = base_dir or Path(__file__).resolve().parent
    created_dirs: list[str] = []
    created_files: list[str] = []

    for d in DIRS:
        try:
            if _ensure_dir(d, base):
                created_dirs.append(str(d))
        except FileExistsError as e:
            raise RuntimeError(str(e)) from e

    for rel_path, content in FILES.items():
        try:
            if _ensure_file(rel_path, content, base):
                created_files.append(rel_path)
        except FileExistsError as e:
            raise RuntimeError(str(e)) from e

    return {"created_dirs": created_dirs, "created_files": created_files}


def main() -> None:
    """Run scaffolding and report results."""
    print("Scaffolding project structure (skipping existing paths)...")
    result = scaffold()
    created_dirs = result["created_dirs"]
    created_files = result["created_files"]

    if created_dirs:
        print("\nCreated directories:")
        for p in created_dirs:
            print(f"  + {p}")
    else:
        print("\nNo new directories created (all already exist).")

    if created_files:
        print("\nCreated files:")
        for p in created_files:
            print(f"  + {p}")
    else:
        print("\nNo new files created (all already exist).")

    if not created_dirs and not created_files:
        print("\nStructure is already complete. Nothing overwritten.")
    else:
        print(f"\nDone. Created {len(created_dirs)} dir(s), {len(created_files)} file(s).")


if __name__ == "__main__":
    main()
