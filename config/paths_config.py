import os

############## Data Ingestion Paths ##############

RAW_DIR = "artifacts/raw"
RAW_FILE_PATH = os.path.join(RAW_DIR, "raw.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR, "test.csv")

CONFIG_PATH = "config/config.yaml"


############## Data Processing Paths ##############

PROCESSED_DIR = "artifacts/processed"
PROCESSED_TRAIN_FILE_PATH = os.path.join(PROCESSED_DIR, "Processed_train.csv")
PROCESSED_TEST_FILE_PATH = os.path.join(PROCESSED_DIR, "Processed_test.csv")


############## Model Training Paths ##############

MODEL_OUTPUT_PATH = "artifacts/models/lgbm_model.pkl"