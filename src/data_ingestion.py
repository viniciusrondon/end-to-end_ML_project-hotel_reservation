import os
import shutil
import sys
from pathlib import Path

import pandas as pd
# from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logging_config import get_logger
from src.exceptions import ProjectError
from config.paths_config import *
from utils.common_functions import read_yaml
import kagglehub

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self,config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_name = self.config["bucket_file_name"]
        self.train_ratio = self.config["train_ratio"]
        self.test_ratio = self.config["test_ratio"]

        os.makedirs(RAW_DIR, exist_ok=True)

        logger.info(f"Data ingestion started with: {self.bucket_name} and file is {self.file_name}")

    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            blob.download_to_filename(RAW_FILE_PATH)
            logger.info(f"CSV file downloaded from {self.bucket_name} and file is {self.file_name}")
        except Exception as e:
            logger.error(f"Error downloading CSV file from {self.bucket_name} and file is {self.file_name}: {e}")
            raise ProjectError("Error downloading CSV file from GCP", sys) from e
        
    def split_data(self):
        try:
            logger.info(f"Splitting data into train and test sets with test_ratio {self.test_ratio}")
            data = pd.read_csv(RAW_FILE_PATH)

            train_data, test_data = train_test_split(data, test_size=self.test_ratio, random_state=42)
            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            test_data.to_csv(TEST_FILE_PATH, index=False)
            logger.info(f"Train and test data saved successfully to {TRAIN_FILE_PATH} and {TEST_FILE_PATH}")
        except Exception as e:
            logger.error(f"Error splitting data into train and test sets: {e}")
            raise ProjectError("Error splitting data into train and test sets", sys) from e

    def run(self):
        try:
            logger.info(f"Starting data ingestion pipeline")
            self.download_csv_from_gcp()
            self.split_data()
            logger.info(f"Data ingestion pipeline completed successfully")
        except Exception as e:
            logger.error(f"Error in data ingestion pipeline: {e}")
            raise ProjectError("Error in data ingestion pipeline", sys) from e
        

class KaggleDataIngestion:
    def __init__(self,config):
        self.config = config["data_ingestion"]
        self.kaggle_url = self.config["kaggle_url"]
        self.kaggle_file_name = self.config["kaggle_file_name"]
        self.train_ratio = self.config["train_ratio"]
        self.test_ratio = self.config["test_ratio"]

        os.makedirs(RAW_DIR, exist_ok=True)
        logger.info(f"Kaggle data ingestion started with: {self.kaggle_url} and file is {self.kaggle_file_name}")

    def _resolve_csv_path(self, downloaded_path: str) -> Path:
        """Resolve the CSV path returned by kagglehub (file or dataset directory)."""
        path = Path(downloaded_path)
        if path.is_file():
            return path

        direct = path / self.kaggle_file_name
        if direct.exists():
            return direct

        matches = list(path.rglob("*.csv"))
        if not matches:
            raise FileNotFoundError(f"No CSV file found under {path}")

        for candidate in matches:
            if candidate.name == self.kaggle_file_name:
                return candidate
        return matches[0]

    def download_csv_from_kaggle(self):
        try:
            downloaded_path = kagglehub.dataset_download(self.kaggle_url)
            source_csv = self._resolve_csv_path(downloaded_path)
            shutil.copy2(source_csv, RAW_FILE_PATH)
            logger.info(
                f"CSV copied from Kaggle dataset {self.kaggle_url} "
                f"({source_csv.name}) to {RAW_FILE_PATH}"
            )
        except Exception as e:
            logger.error(
                f"Error downloading CSV from Kaggle dataset {self.kaggle_url}: {e}. "
                "Ensure Kaggle API credentials are configured and you accepted the dataset terms."
            )
            raise ProjectError("Error downloading CSV file from Kaggle", sys) from e

    def split_data(self):
        try:
            logger.info(f"Splitting data into train and test sets with test_ratio {self.test_ratio}")
            data = pd.read_csv(RAW_FILE_PATH)
            train_data, test_data = train_test_split(data, test_size=self.test_ratio, random_state=42)
            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            test_data.to_csv(TEST_FILE_PATH, index=False)
            logger.info(f"Train and test data saved successfully to {TRAIN_FILE_PATH} and {TEST_FILE_PATH}")
        except Exception as e:
            logger.error(f"Error splitting data into train and test sets: {e}")
            raise ProjectError("Error splitting data into train and test sets", sys) from e

    def run(self):
        try:
            logger.info(f"Starting kaggle data ingestion pipeline")
            self.download_csv_from_kaggle()
            self.split_data()
            logger.info(f"Kaggle data ingestion pipeline completed successfully")
        except ProjectError:
            raise
        except Exception as e:
            logger.error(f"Error in kaggle data ingestion pipeline: {e}")
            raise ProjectError("Error in kaggle data ingestion pipeline", sys) from e


if __name__ == "__main__":
    config = read_yaml(CONFIG_PATH)
    data_ingestion = KaggleDataIngestion(config)
    data_ingestion.run()