"""
Author: Amel Sellami 
Date creation: 22-04-2022

copy the files model, data trace and f1 score in the deployement folder.
"""
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import shutil
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

ingestion_record = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path'])


####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    shutil.copy(
        os.path.join(
            ingestion_record,
            'ingestedfiles.txt'),
        prod_deployment_path)
    shutil.copy(
        os.path.join(
            model_path,
            'trainedmodel.pkl'),
        prod_deployment_path)
    shutil.copy(
        os.path.join(
            model_path,
            'latestscore.txt'),
        prod_deployment_path)
        
        
        
if __name__ == "__main__":
    store_model_into_pickle()  

