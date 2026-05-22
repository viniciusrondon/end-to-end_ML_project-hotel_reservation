import os
import sys
import pandas as pd
import yaml
from src.exceptions import ProjectError
from src.logging_config import get_logger

logger = get_logger(__name__)

def read_yaml(file_path: str) -> dict:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist")
        with open(file_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {file_path} loaded successfully")
            return config
    except Exception as e:
        logger.error(f"Error reading yaml file: {e}")
        raise ProjectError("Failed to read yaml file", sys) from e

def load_data(path: str):
    try:
        logger.info(f"Loading data from {path}")
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error loading data from {path}: {e}")
        raise ProjectError("Failed to load data", sys) from e

