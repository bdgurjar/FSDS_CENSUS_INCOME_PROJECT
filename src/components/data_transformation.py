import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd

from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OneHotEncoder # One hot encoding Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
                    
            
            # Define which columns should be onehot-encoded and which should be scaled
            numerical_cols=["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
            category_cols=["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]
            
            # Define onehot encoded variable
            workclass_cat=[' State-gov', ' Self-emp-not-inc', ' Private', ' Federal-gov',' Local-gov', ' others', ' Self-emp-inc', ' Without-pay', ' Never-worked']
            education_cat=[' Bachelors', ' HS-grad', ' 11th', ' Masters', ' 9th',' Some-college', ' Assoc-acdm', ' Assoc-voc', ' 7th-8th',' Doctorate', ' Prof-school', ' 5th-6th', ' 10th', ' 1st-4th',' Preschool', ' 12th']
            marital_status_cat=[' Never-married', ' Married-civ-spouse', ' Divorced',' Married-spouse-absent', ' Separated', ' Married-AF-spouse',' Widowed']
            occupation_cat=[' Adm-clerical', ' Exec-managerial', ' Handlers-cleaners',' Prof-specialty', ' Other-service', ' Sales', ' Craft-repair',' Transport-moving', ' Farming-fishing', ' Machine-op-inspct',' Tech-support', ' others', ' Protective-serv', ' Armed-Forces',' Priv-house-serv']
            relationship_cat=[' Not-in-family', ' Husband', ' Wife', ' Own-child', ' Unmarried',' Other-relative']
            race_cat=[' White', ' Black', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo',' Other']
            sex_cat=[' Male', ' Female']
            native_country_cat=[' United-States', ' Cuba', ' Jamaica', ' India', ' others',' Mexico', ' South', ' Puerto-Rico', ' Honduras', ' England',' Canada', ' Germany', ' Iran', ' Philippines', ' Italy',' Poland', ' Columbia', ' Cambodia', ' Thailand', ' Ecuador',' Laos', ' Taiwan', ' Haiti', ' Portugal', ' Dominican-Republic',' El-Salvador', ' France', ' Guatemala', ' China', ' Japan',' Yugoslavia', ' Peru', ' Outlying-US(Guam-USVI-etc)', ' Scotland',' Trinadad&Tobago', ' Greece', ' Nicaragua', ' Vietnam', ' Hong',' Ireland', ' Hungary', ' Holand-Netherlands']
                        
            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='mean')),
                ('scaler',StandardScaler())

                ]

            )
            #Categorical onehote encoder pipeline
            cat_onehot_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ("onehotencoder",OneHotEncoder(categories=[workclass_cat,education_cat,marital_status_cat,occupation_cat,relationship_cat,race_cat,sex_cat,native_country_cat],sparse_output=False)),
                ("scaler",StandardScaler())
                ]

            )



            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ("cat_onehot_pipeline",cat_onehot_pipeline,category_cols)

            ])
            
            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        
    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            #train_df["education_num"]=train_df["education_num"].astype("str")
            #test_df["education_num"]=test_df["education_num"].astype("str")
            
            target_column_name = 'Income'
            drop_columns = [target_column_name]

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)
