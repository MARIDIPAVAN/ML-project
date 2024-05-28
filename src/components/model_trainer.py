import sys, os
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '../..')))

# Import libraries
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj, model_evaluator
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Create a ModelTrainerConfig class
@dataclass
class ModelTrainerConfig:
    model_file_path: str = os.path.join(os.getcwd(), "artifacts", "model.pkl")

# Create a ModelTrainer class
class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()
        
    def initiateModelTrainer(self, clean_train_arr, clean_test_arr):
        logging.info('Initiating Model Trainer')
        
        try:
            # Split the data
            X_train, y_train, X_test, y_test = (
                clean_train_arr[:, :-1],
                clean_train_arr[:, -1],
                clean_test_arr[:, :-1],
                clean_test_arr[:, -1],
            )
            logging.info('Data successfully split')
            
            # Specify the models
            models = {
                'LogisticRegression': LogisticRegression(),
                'DecisionTreeClassifier': DecisionTreeClassifier(),
                'RandomForestClassifier': RandomForestClassifier(),
                'GradientBoostingClassifier': GradientBoostingClassifier()
            }
            
            # Get the best model
            model_report = model_evaluator(X_train, y_train, X_test, y_test, models)
            
            # Sort based on the F1 score
            sorted_model_report = sorted(model_report.items(), key=lambda x: x[1]['f1'], reverse=True)
            
            logging.info(f'Sorted model report is {sorted_model_report}')
            best_model_name, best_score = sorted_model_report[0]
            best_model = models[best_model_name]
            logging.info(f'Best model is {best_model_name} with score {best_score}')
            
            # Train the best model on the entire training data
            best_model.fit(X_train, y_train)
            
            # Save the model
            save_obj(self.trainer_config.model_file_path, best_model)
            logging.info('Model has been saved successfully')
            
        except Exception as e:
            logging.error(f'Error occurred during training the model: {e}')
            raise CustomException(e, sys)
