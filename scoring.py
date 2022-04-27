"""
Author: Amel Sellami 
Date creation: 22-04-2022

This file consists of modules that are needed to get the f1 score of the model. 
"""
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    
    # read test data
    test_data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    # read the save model
    model = pickle.load(open(os.path.join(model_path,'trainedmodel.pkl'),'rb'))
    
    # calculate f1 score
    
    label = test_data.pop('exited')
    x_test = test_data.drop(['corporation'], axis=1)
    
    preds = model.predict(x_test)
    f1 = metrics.f1_score(preds, label)
    
    with open(os.path.join(model_path, 'latestscore.txt'), 'w') as fp:
        fp.write(f"F1 score: {f1}")
   
   
if __name__ == "__main__":
    score_model()  

