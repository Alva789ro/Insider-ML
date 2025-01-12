import xgboost as xgb
import boto3
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from json import loads

S3_BUCKET = "" #your s3 bucket
MODEL_KEY = "" 

def prepare_data(data):
    """
    Preparing data for predictions
    #TODO: implement to handle multiple preds at once.
    """
    data = pd.DataFrame(loads(data), index=[0])
    #encodings for text data
    for col in ['Title', 'Trade Type']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
    data = pd.DataFrame(data, index = [0]).rename(columns={"Trade_Type": "Trade Type"})
    return data

def load_xgbInsider():
    """
    Load model either from local path or from S3.
    """
    #loading from S3
    s3 = boto3.client("s3")
    local_model_path = "/tmp/xgboost_model.json"
    s3.download_file(S3_BUCKET, MODEL_KEY, local_model_path)

    loaded_model = xgb.XGBClassifier()
    loaded_model.load_model(local_model_path)
    print("Properly loaded xgbInsider")
    return loaded_model

def predict(model, input_data):
    # model = load_xgbInsider()
    preds = model.predict(input_data)
    preds = "positive" if preds[0] == 1 else "negative"
    
    return preds