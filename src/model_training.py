import os
import sys
import pandas as pd
import joblib
from src.logging_config import get_logger
from src.exceptions import ProjectError
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml, load_data
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import randint, uniform

import mlflow

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, train_path: str, test_path: str, model_output_path: str):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_distribution = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_and_split_data(self):
        try:
            logger.info(f"Loading and splitting data from {self.train_path}")
            train_data = load_data(self.train_path)

            logger.info(f"Loading test data from {self.test_path}")
            test_data = load_data(self.test_path)

            X_train = train_data.drop(columns=["booking_status"])
            y_train = train_data["booking_status"]
            X_test = test_data.drop(columns=["booking_status"])
            y_test = test_data["booking_status"]
            logger.info(f"Data loaded and split successfully")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f"Error loading and splitting data: {e}")
            raise ProjectError("Error loading and splitting data", sys) from e
    
    def train_lgbm(self, X_train, y_train):
        try:
            logger.info(f"Training LightGBM model")
            lgbm = LGBMClassifier(random_state=self.random_search_params["random_state"])
            random_search = RandomizedSearchCV(
                estimator=lgbm,
                param_distributions=self.params_distribution,
                n_iter=self.random_search_params["n_iter"],
                cv=self.random_search_params["cv"],
                verbose=self.random_search_params["verbose"],
                random_state=self.random_search_params["random_state"],
                scoring=self.random_search_params["scoring"],
                n_jobs=self.random_search_params["n_jobs"]
            )

            logger.info(f"Random search parameters initiated")
            random_search.fit(X_train, y_train)

            logger.info(f"Hyperparameter tuning completed")
            best_lgbm = random_search.best_estimator_
            logger.info(f"Best LightGBM model parameters: {random_search.best_params_}")

            logger.info(f"LightGBM model trained and tuned successfully")
            return best_lgbm
        except Exception as e:
            logger.error(f"Error training LightGBM model: {e}")
            raise ProjectError("Error training LightGBM model", sys) from e
    
    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info(f"Evaluating LightGBM model")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"Precision: {precision}")
            logger.info(f"Recall: {recall}")
            logger.info(f"F1 Score: {f1}")
            logger.info(f"LightGBM model evaluated successfully")
            return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
        except Exception as e:
            logger.error(f"Error evaluating LightGBM model: {e}")
            raise ProjectError("Error evaluating LightGBM model", sys) from e

    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            logger.info(f"Saving model")
            joblib.dump(model, self.model_output_path)
            logger.info(f"Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving LightGBM model: {e}")
            raise ProjectError("Error saving LightGBM model", sys) from e
    
    def run(self):
        try:
            with mlflow.start_run():
                logger.info(f"Starting model training pipeline")

                logger.info(f"Starting mlflow run")

                logger.info(f"Logging the training and testing dataset to MLFlow")
                mlflow.log_artifact(self.train_path, artifact_path="dataset")
                mlflow.log_artifact(self.test_path, artifact_path="dataset")

                X_train, y_train, X_test, y_test = self.load_and_split_data()
                best_lgbm = self.train_lgbm(X_train, y_train)
                metrics = self.evaluate_model(best_lgbm, X_test, y_test)
                self.save_model(best_lgbm)

                logger.info(f"Logging the model to MLFlow")
                mlflow.log_artifact(self.model_output_path, artifact_path="model")

                logger.info(f"Logging the model parameters to MLFlow")
                mlflow.log_params(best_lgbm.get_params())

                logger.info(f"Logging the metrics to MLFlow")
                mlflow.log_metrics(metrics)

                logger.info(f"Model training completed successfully")
        except Exception as e:
            logger.error(f"Error running model training: {e}")
            raise ProjectError("Error running model training", sys) from e

if __name__ == "__main__":
    model_training = ModelTraining(PROCESSED_TRAIN_FILE_PATH, PROCESSED_TEST_FILE_PATH, MODEL_OUTPUT_PATH)
    model_training.run()