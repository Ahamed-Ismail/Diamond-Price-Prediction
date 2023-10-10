import os, sys
import pickle
import pandas as pd
import numpy as np

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.logger import logging
from src.exception import CustomException

def save_obj(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path , exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    
    except Exception as e:
        logging.info("Error occured at Save_obj in utils.py")
        raise CustomException(e,sys)

def evaluate_model(models, xtrain,ytrain,xtest,ytest):

    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            model.fit(xtrain,ytrain)

            #Make Predictions
            ypred=model.predict(xtest)

            #rmse = np.sqrt(mean_squared_error(ytest, ypred))
            r2 = r2_score(ytest, ypred)

            print(list(models.keys())[i])
            report[(list(models.keys())[i])]=[r2]

            return report
        
    except Exception as e:
        logging(f"Error occured at evaluate_model in utils.py")
        raise CustomException(e,sys)
    
def load_object(obj_path):

    try:
        with open(obj_path, 'rb') as obj:
            return pickle.load(obj)

    except Exception as e:
        logging(f"Error occured at load_object in utils.py")
        raise CustomException(e,sys)