"""
Author: Amel Sellami 
Date creation: 22-04-2022

This file consists of modules that are required for doing a complete diagnosis 
of the training.
The modules are model prediction, summary stats, percentile of missign values
timing of important ml tasks and that dependancies are up-to-date.

"""
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path']) 
data_summary =   os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
output_dir = os.path.join(config['prod_deployment_path']) 


##################Function to get model predictions
def model_predictions():
    #read the deployed model and a test dataset, calculate predictions
    test_data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    test_data.drop(['corporation', 'exited'], axis = 1, inplace = True)
    # read deployed model
    with open(os.path.join(model_path, 'trainedmodel.pkl') , 'rb') as f:
        model = pickle.load(f)
    
    preds = model.predict(test_data)
    assert len(test_data) == len(preds)
    
    return preds 
 
##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics mean, median and std
    # generate the results in a dict
    data = pd.read_csv(os.path.join(data_summary, 'finaldata.csv'))
    data.drop(['corporation', 'exited'], axis = 1, inplace = True)
    stats = []
    for col in data.columns: 
        stats.append([col, "mean",  data[col].mean()])
        stats.append([col, "median",  data[col].median()])
        stats.append([col, "std",  data[col].std()])

    return  stats #return value should be a list containing all summary statistics

##################Function to get missign values in data
def missing_vals():
    # returns the percentile of missign values by column
    test_data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    
    missing = []
    for col in test_data.columns:
        count_na = test_data[col].isna().sum()
        missing.append([col, str(int(count_na/((count_na + len(test_data))*100)))])
    return missing


##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    
    result = []
    for procedure in ["training.py" , "ingestion.py"]:
        starttime = timeit.default_timer()
        os.system('python %s' % procedure)
        timing=timeit.default_timer() - starttime
        result.append([procedure, timing])
    return str(result) #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of packages
    process = subprocess.run(
        "pip list --outdated --format columns",
        shell=True,
        capture_output=True,
        check=True,
        text=True
    )

    outdated_packages_file = os.path.join(
        output_dir, "outdated_packages.txt"
    )
    with open(outdated_packages_file, 'w') as f:
        print(f"Writing list of outdated packages to {outdated_packages_file}")
        f.write(str(process.stdout))
    return str(process.stdout)



if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    missing_vals()
    execution_time()
    outdated_packages_list()
