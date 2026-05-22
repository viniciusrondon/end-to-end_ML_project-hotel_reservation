import json
import os
from pathlib import Path
import pandas as pd
import sys
import numpy as np
from src.logging_config import get_logger
from src.exceptions import ProjectError
from config.paths_config import *
from utils.common_functions import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:
    def __init__(self,train_path: str, test_path: str, processed_dir: str,config_path: str):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir, exist_ok=True)

    def preprocess_data(self, df):
        try:
            logger.info(f"Preprocessing data step")

            logger.info(f"Dropping Booking_ID column")
            df.drop(columns=["Booking_ID"], inplace=True)

            logger.info(f"Dropping duplicate rows")
            df.drop_duplicates(inplace=True)

            logger.info(f"Getting categorical and numerical columns from config, applying label encoding")
            categorical_columns = self.config["data_processing"]["categorical_columns"]
            numerical_columns = self.config["data_processing"]["numerical_columns"]

            label_encoder = LabelEncoder()

            mappings={}
            # encoder_data = df.copy()

            for col in categorical_columns:
                df[col] = label_encoder.fit_transform(df[col])
                mappings[col]={label:code for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
            
            # try:
            #     logger.info(f"Saving mappings to {self.processed_dir}")
            #     with open(os.path.join(self.processed_dir, "mappings.json"), "w") as f:
            #         json.dump(mappings, f, indent=4)
            #     logger.info(f"Mappings saved successfully to {os.path.join(self.processed_dir, "mappings.json")}")
            # except Exception as e:
            #     logger.error(f"Error saving mappings to {os.path.join(self.processed_dir, "mappings.json")}: {e}")
            #     raise ProjectError("Error saving mappings", sys) from e
            
            logger.info("Skewness handling")
            skewness = df.skew()
            for col in df.columns:
                if skewness[col] > self.config["data_processing"]["skewness_threshold"]:
                    df[col] = np.log1p(df[col])
            logger.info("Skewness handled successfully")

            return df
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise ProjectError("Error preprocessing data", sys) from e
    
    def balance_data(self, df):
        try:
            logger.info(f"Balancing data step")
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(df.drop(columns=["booking_status"]), df["booking_status"])
            balanced_df = pd.DataFrame(X_res, columns=df.columns[:-1])
            balanced_df["booking_status"] = y_res
            logger.info("Data balanced successfully")
            return balanced_df
        except Exception as e:
            logger.error(f"Error balancing data: {e}")
            raise ProjectError("Error balancing data", sys) from e

    def select_features(self, df):
        try:
            logger.info(f"Selecting features step")
            X = df.drop(columns=["booking_status"])
            y = df["booking_status"]
            model = RandomForestClassifier(random_state=42)
            model.fit(X,y)
            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                "Feature": X.columns,
                "importance": feature_importance
            })
            top_features_importance_df = feature_importance_df.sort_values(by="importance", ascending=False)
            top_features = top_features_importance_df["Feature"].head(self.config["data_processing"]["feature_selection_threshold"]).values
            top_df = df[top_features.tolist() + ["booking_status"]]
            return top_df
        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            raise ProjectError("Error selecting features", sys) from e
    
    def save_data(self, df, file_path: str):
        try:
            logger.info(f"Saving data processed in processed folder")
            df.to_csv(file_path, index=False)
            logger.info(f"Data saved successfully to {file_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise ProjectError("Error saving processed data", sys) from e

    def run(self):
        try:
            logger.info(f"Starting data preprocessing pipeline from RAW directory")
            train_data = load_data(self.train_path)
            test_data = load_data(self.test_path)
            logger.info(f"Train and Test data loaded successfully")

            logger.info(f"Preprocessing train and test data")
            train_data = self.preprocess_data(train_data)
            test_data = self.preprocess_data(test_data)
            logger.info(f"Preprocessing train and test data completed successfully")

            logger.info(f"Balancing train data")
            train_data = self.balance_data(train_data)
            logger.info(f"Balancing train data completed successfully")

            logger.info(f"Selecting features for train data")
            train_data = self.select_features(train_data)
            test_data = test_data[train_data.columns]
            logger.info(f"Selecting features for train and test data completed successfully")

            logger.info(f"Saving train data to processed folder")
            self.save_data(train_data, PROCESSED_TRAIN_FILE_PATH)
            self.save_data(test_data, PROCESSED_TEST_FILE_PATH)
            logger.info(f"Saving train and test data to processed folder completed successfully")
        except Exception as e:
            logger.error(f"Error in data preprocessing pipeline: {e}")
            raise ProjectError("Error in data preprocessing pipeline", sys) from e

if __name__ == "__main__":
    data_processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    data_processor.run()