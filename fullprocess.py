"""
Author: Amel Sellami 
Date creation: 27-04-2022

full  procss file.
"""
import json
import os
import glob

from deployment import store_model_into_pickle as deploy
import apicalls as make_api_calls
from ingestion import merge_multiple_dataframe as ingest
from reporting import confusion_matrix as report
from scoring import score_model as score
from training import train_model as train

with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
deployment_path = config['prod_deployment_path']
model_path = config['output_model_path']
ingestion_record = os.path.join(deployment_path, 'ingestedfiles.txt')

def check_new_files(dataset_dir, ingestion_record):
    """
    Check for files in dataset_dir not in ingestion_record
    :param dataset_dir: Directory containing CSV datasets
    :param ingestion_record: File containing list of already ingested datasets
    :return: True if new files exist in dataset_dir else False
    """
    print('Checking if there are any new files!')
    with open(ingestion_record, 'r') as f:
        ingested_files = f.read().splitlines()

    candidate_files = glob.glob(f'{dataset_dir}/*.csv')
    new_files = [f for f in candidate_files if f not in ingested_files]
    return bool(new_files)


def check_model_drift(metric_file):
    """
    Train and score new model on latest finaldata_*.csv.
    Compare new F1-Score to old one.
    :param metric_file: File containing old F1-Score
    :return: New F1-Score > Old F1-Score, new F1-Score, old F1-Score
    """
    print('Checking for model drift!')
    with open(metric_file, 'r') as f:
        s = f.readline().strip()
        st = ''.join(x for x in s if x.isdigit())
        old_f1_score = float(st)

    train()
    new_f1_score = score()

    return new_f1_score > old_f1_score, new_f1_score, old_f1_score

def main():

    if not check_new_files(input_folder_path, ingestion_record):
        print(f'No new dataset in {input_folder_path}. Ending process...')
        exit()

    ingest()

    metric_file = os.path.join(deployment_path, 'latestscore.txt')
    drift, new_score, old_score = check_model_drift(metric_file)
    if not drift:
        print('Production model performs better than the deployed one.'
              f'New F1-Score: {new_score}. Old F1-Score: {old_score}')
        exit()

    deploy()
    report()
    make_api_calls()


if __name__ == '__main__':
    main()
