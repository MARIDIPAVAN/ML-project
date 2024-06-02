import sys, os
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '../..')))

# Import libraries
import os, sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj
from dataclasses import dataclass
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer  # Handling Missing Values
from sklearn.preprocessing import OrdinalEncoder  # Ordinal Encoding
from sklearn.feature_extraction.text import TfidfVectorizer  # Text Vectorization
# pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_obj
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_file_path: str = os.path.join(os.getcwd(), "artifacts", "preprocessor.pkl")
    clean_train_file_path: str = os.path.join(os.getcwd(), "artifacts", "clean_train.csv")
    clean_test_file_path: str = os.path.join(os.getcwd(), "artifacts", "clean_test.csv")

class DataTransformation:
    
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        
    def getPreprocessorObject(self):
        # Define the categorical and text columns
    
        text_col = 'message'
        
       

        # Text Pipeline
        text_pipeline = Pipeline(
            steps=[
                ('tfidf', TfidfVectorizer())
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('text_pipeline', text_pipeline, text_col)
            ]
        )
        
        logging.info('Preprocessor created successfully!')
        return preprocessor
    
    def initiateDataTransformation(self, train_path, test_path):
        logging.info('Data Transformation has started')
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            logging.info('Train data read successfully')
            
            test_df = pd.read_csv(test_path)
            logging.info('Test data read successfully')
            
            # Debug: Print the columns to check if 'label' and 'message' exist
            #print("Train DataFrame Columns:", train_df.columns)
            #print("Test DataFrame Columns:", test_df.columns)
            
            # Split dependent and independent features
            if 'label' not in train_df.columns or 'message' not in train_df.columns:
                raise KeyError("Columns 'label' and 'message' must be present in the data")

            X_train = train_df[['label', 'message']]
            X_test = test_df[['label', 'message']]
            y_train = train_df['label']
            y_test = test_df['label']
            logging.info('Splitting of dependent and independent features is successful')
            
            # Get preprocessor and preprocess the content
            preprocessor = self.getPreprocessorObject()
            X_train_arr = preprocessor.fit_transform(X_train)
            logging.info('X_train successfully pre-processed')
            
            X_test_arr = preprocessor.transform(X_test)
            logging.info('X_test successfully pre-processed')
            
            # Ensure y_train and y_test are numpy arrays
            y_train = y_train.to_numpy().reshape(-1, 1)  # Ensure y_train is 2D for concatenation
            y_test = y_test.to_numpy().reshape(-1, 1)    # Ensure y_test is 2D for concatenation

            # Ensure X_train_arr and X_test_arr are dense numpy arrays
            if hasattr(X_train_arr, "toarray"):
                X_train_arr = X_train_arr.toarray()
            if hasattr(X_test_arr, "toarray"):
                X_test_arr = X_test_arr.toarray()

            clean_train_arr = np.hstack((X_train_arr, y_train))
            clean_test_arr = np.hstack((X_test_arr, y_test))
            logging.info('Concatenation of cleaned arrays is successful')
            
            # Save the preprocessor 
            save_obj(self.transformation_config.preprocessor_file_path, preprocessor)
            logging.info('Pre-processor successfully saved')
            
            return clean_train_arr, clean_test_arr
            
        except Exception as e:
            logging.error(f'Exception occurred in Data Transformation: {e}')
            raise CustomException(e, sys)

# Example usage
train_path = "C:/Users/pavan/projects/spam classifier/artifacts/train.csv"
test_path = "C:/Users/pavan/projects/spam classifier/artifacts/test.csv"
data_transformation = DataTransformation()
train_arr,test_arr=data_transformation.initiateDataTransformation(train_path, test_path)

modeltrainer=ModelTrainer()
modeltrainer.initiateModelTrainer(train_arr,test_arr)
