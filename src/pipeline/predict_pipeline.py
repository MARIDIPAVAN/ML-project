import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_obj
import logging

class PredictPipeline:
    def __init__(self):
        # Load model and preprocessor
        try:
            self.model_path = os.path.join("artifacts", "model.pkl")
            self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            logging.info("Loading model and preprocessor")
            self.model = load_obj(file_path=self.model_path)
            self.preprocessor = load_obj(file_path=self.preprocessor_path)
            logging.info("Model and preprocessor loaded successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features):
        try:
            logging.info(f"Features before preprocessing: {features}")
            data_scaled = self.preprocessor.transform(features)
            logging.info(f"Features after preprocessing: {data_scaled}")

            preds = self.model.predict(data_scaled)
            logging.info(f"Predictions: {preds}")
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, message: str):
        if message is None:
            raise ValueError("The 'message' cannot be None")
        self.message = message

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "message": [self.message]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)

