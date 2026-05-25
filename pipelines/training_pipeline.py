import sys
from src.logging_config import get_logger
from src.exceptions import ProjectError
from config.paths_config import *
from utils.common_functions import read_yaml
from src.data_ingestion import KaggleDataIngestion
from src.data_preprocessing import DataProcessor
from src.model_training import ModelTraining

logger = get_logger(__name__)

if __name__ == "__main__":
    try:
        logger.info(f"Starting training pipeline")

        logger.info(f"Data ingestion started")
        config = read_yaml(CONFIG_PATH)
        data_ingestion = KaggleDataIngestion(config)
        data_ingestion.run()

        logger.info(f"Data preprocessing started")
        data_processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
        data_processor.run()

        logger.info(f"Model training started")
        model_training = ModelTraining(PROCESSED_TRAIN_FILE_PATH, PROCESSED_TEST_FILE_PATH, MODEL_OUTPUT_PATH)
        model_training.run()

        logger.info(f"Training pipeline completed successfully")
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise ProjectError("Error in training pipeline", sys) from e