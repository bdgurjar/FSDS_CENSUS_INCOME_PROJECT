import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging


def whitespace_remover(dataframe):
    for i in dataframe.columns:
        if dataframe[i].dtype=='object':
            dataframe[i]=dataframe[i].map(str.strip)
        else:
            pass
    return dataframe

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def model_evalute(X_train, y_train, X_test, y_test,models):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            # Train Model
            model.fit(X_train,y_train)

            # Predict Testing Data
            y_pred=model.predict(X_test)

            # Get accuracy scores for train and test data
            test_model_score=accuracy_score(y_test,y_pred)

            report[(list(models.keys())[i])]  = test_model_score
            
        return report

    except Exception as e:
        logging.info("Exception Occured during Model Training")
        raise CustomException(e,sys)


