import os
import sys
sys.path.append(r'c:\Users\tanji\Desktop\myPW\end to endprojects\DiamondPricePrediction')

from src.exception import CustomException
from src.logger import logging

import pandas as pd

from src.components.data_ingestion import DataIngestion

if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    print(train_data_path,test_data_path)
