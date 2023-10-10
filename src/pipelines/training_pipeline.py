import os
import sys
sys.path.append(r'c:\Users\tanji\Desktop\myPW\end to endprojects\DiamondPricePrediction')

from src.exception import CustomException
from src.logger import logging

import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__=='__main__':
    data_injestion_obj=DataIngestion()
    train_data_path,test_data_path=data_injestion_obj.initiate_data_ingestion()
    print(train_data_path,test_data_path)

    data_transformation_obj=DataTransformation()
    train_arr, test_arr, obj_path=data_transformation_obj.initiate_data_transformation(train_data_path, test_data_path)

    model_trainer=ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr,test_arr)