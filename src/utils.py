import sys,os
from os.path import dirname,join,abspath
sys.path.insert(0,abspath(join(dirname(__file__),'..')))

from src.logger import logging
from src.exception import CustomException

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

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