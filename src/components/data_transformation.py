from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import sys,os
from dataclasses import dataclass
from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    preproceesor_obj_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformation_object(self):

        try:
            logging.info("Data transfer initiated")

            #categorical and numerical columns
            cat_col=['cut','color','clarity']
            num_col=['carat','depth','table','x','y','z']

            #assigning ranks to categorical features
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("Data transfer pipeline initiated")

            #numerical pipeline
            num_pipeline=Pipeline(
                steps=[
                        ('imputer',SimpleImputer(strategy='median')),
                        ('scaler',StandardScaler())
                ]
            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories],handle_unknown='use_encoded_value')),
                    ('scaler',StandardScaler())
                ]
            )

            #columntransformer
            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,num_col),
                ('cat_pipeline',cat_pipeline,cat_col)
                ]
            )

            logging.info("Data transformation object completed")

            return preprocessor

        except Exception as e:
            logging.info("Exception occured in data_transformation.py")
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self, train_data_path, test_data_path):

        try:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)

            logging.info("Read train and test completed")
            logging.info(f"Train Head\n{train_df.head().to_string()}")
            logging.info(f"Test Head\n{test_df.head().to_string()}")

            logging.info("Getting Preprocessing object")

            processing_obj=self.get_data_transformation_object()

            target_column='price'

            drop_column=['id',target_column]

            #training data
            input_feature_train_df=train_df.drop(drop_column, axis=1)
            target_train_df=train_df[target_column]

            #testing data
            input_feature_test_df=test_df.drop(drop_column, axis=1)
            target_test_df=test_df[target_column]

            #processing data
            input_feature_train_arr=processing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=processing_obj.transform(input_feature_test_df)

            #concat columns

            train_arr=np.c_[input_feature_train_arr,np.array(target_train_df)]
            test_arr=np.c_[input_feature_test_arr, np.array(target_test_df)]

            #saving preproceesor

            save_obj(
                file_path=self.data_transformation_config.preproceesor_obj_path,
                obj=processing_obj
            )


            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preproceesor_obj_path
            )

        except Exception as e:
            logging.info("Exception occured in data_transformation.py")
            raise CustomException(e,sys)