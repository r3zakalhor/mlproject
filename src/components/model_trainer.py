import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, r2_score
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        logging.info("Model Trainer started")
        try:
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "SVR": SVR(),
                "KNeighbors": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            logging.info(f"Best model found: {best_model_name} with score {best_model_score}")

            if best_model_score < 0.6:
                logging.error("No suitable model found with score above threshold")
                raise CustomException("No suitable model found", sys)
            
            #preprocessor_obj = pd.read_pickle(preprocessor_path)
            #logging.info("Combining preprocessor and best model into a single pipeline")
            #model_pipeline = pd.pipeline.Pipeline(steps=[
            #    ('preprocessor', preprocessor_obj),
            #    ('model', best_model)
            #])

            logging.info("Saving the best model")
            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            predictions = best_model.predict(X_test)
            r2_square = r2_score(y_test, predictions)

            return r2_square
        
        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)