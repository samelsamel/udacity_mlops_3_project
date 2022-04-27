"""
Author: Amel Sellami 
Date creation: 21-04-2022

This file consists of modules that are required for training a LR model. 
It train, fit and save the model in a pickle file.

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

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


#################Function for training the model
def train_model():
    """Train LR model."""
    # Read data
    data = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    
    y = data.pop('exited')
    x = data.drop(['corporation'], axis =1)
    # Use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                              intercept_scaling=1, l1_ratio=None, max_iter=100,
                              multi_class='auto', n_jobs=None, penalty='l2',
                              random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                              warm_start=False)
    
    
    # fit the logistic regression to the data
    model.fit(x,y)
    
    #write the trained model to a file called trainedmodel.pkl
    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'wb') as fp:
        pickle.dump(model, fp)

if __name__ == '__main__':
    train_model()