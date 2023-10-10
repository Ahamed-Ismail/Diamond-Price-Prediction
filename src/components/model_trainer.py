import pandas as pd
import numpy as np
import sys,os

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj
from src.utils import evaluate_model
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):

        try:
            logging.info("Spliting into dependent and independent features")

            

            X_train,X_test,y_train,y_test=(
                train_array[:,:-1],
                test_array[:,:-1],
                train_array[:,-1],
                test_array[:,-1]
            )
        
            models={
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'Elasticnet':ElasticNet(),
                'DecisionTree':DecisionTreeRegressor(),
                'RandomForest':RandomForestRegressor(),
                'KNeighbor':KNeighborsRegressor()
            }
            
            model_report:dict=evaluate_model(models, X_train, y_train, X_test, y_test)
            print('\n=============================================================')
            logging.info(f"model report:\n {model_report}")

            best_score=max(list(model_report.values()))
            best_model_name=list(models.keys())[
                list(model_report.values()).index(best_score)
            ]

            best_model=models[best_model_name]
            logging.info(f"Best model found \n {best_model} with r2 score {best_score}")
            print(f"Best model found \n {best_model} with r2 score {best_score}")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info("Error occured at model_trainer.py")
            raise CustomException(e,sys)