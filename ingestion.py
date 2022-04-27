"""
Author: Amel Sellami 
Date creation: 21-04-2022

This file consists of modules that are required for concatenating different datasets: 
merge_multiple_dataframe: help merge different datasets and remove duplicates.

"""
import pandas as pd
import numpy as np
import os
import glob
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


#############Function for data ingestion
def merge_multiple_dataframe():
    
    csv_files = glob.glob("%s/*.csv" % input_folder_path)
    # Load the dataframes
    dfs = (pd.read_csv(p, encoding='utf8') for p in glob.glob("%s/*.csv" % input_folder_path))
    # concat
    res = pd.concat(dfs)
    # remove duplicates
    res = res.drop_duplicates()
    # save result to csv
    res.to_csv("%s/finaldata.csv" % output_folder_path, index = False)
    
    # print the name of the files to ingestedfiles.txt
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'w') as file:
        for line in csv_files:
            file.write(f'time of the ingestion: {datetime.now()}\n')
            file.write(line + '\n')
            

if __name__ == '__main__':
    merge_multiple_dataframe()
