import sys,os
from os.path import dirname,join,abspath
sys.path.insert(0,abspath(join(dirname(__file__),'..')))

from src.logger import logging
from src.exception import CustomException

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pickle as pkl

def save_obj(file_path,obj):
    logging.info(f'initiating to save the file at {file_path}')
    
    try:
        
        #get the directory name
        dirname=os.path.dirname(file_path)
        logging.info(f'Obtained directory name to save the object,{dirname}')
        
        
        #create directory
        os.makedirs(dirname,exist_ok=True)
        logging.info('successfully created the directory')        
        #save the model
        
        with open(file_path,'wb') as f:
            pkl.dump(obj,f)
        f.close()
        
        logging.info('Successfully converted object to a pkl file')
        
    except CustomException as e:
        logging.info(f'Error while saving a obj, {e}')
        print(e)

def load_obj(file_path):
    logging.info('Process of loading object started')
    
    try:
        with open(file_path,'rb') as f:
            obj=pkl.load(f)
        f.close()
        logging.info(f'Object at {file_path} loaded successfully')
        return obj
        
    except CustomException as e:
        logging.info(f'Error occurred while loading object,{e}')
        print(e)

def model_evaluator(X_train, y_train, X_test, y_test, models):
    report = {}
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        report[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return report




