import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import model_evalute
from dataclasses import dataclass


@dataclass
class ModelTrainingConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainingConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting Dependent and Independent variables from train and test array")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
             'LogisticRegression':LogisticRegression(random_state=42),
             'DecisionTreeClassifier':DecisionTreeClassifier(max_depth=6, min_samples_leaf=3, min_samples_split=5,random_state=42)
        }
            model_report:dict=model_evalute(X_train, y_train, X_test, y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from Dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , accuracy_score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , accuracy_score : {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        except Exception as e:
            logging.info("Exception Occured at Model Training")
            raise CustomException(e,sys)