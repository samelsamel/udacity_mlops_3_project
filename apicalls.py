import os
import sys
import requests
import json

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1/"

with open('config.json','r') as f:
    config = json.load(f) 
model_path = os.path.join(config['output_model_path'])

#Call each API endpoint and store the responses
response1 = requests.post("%s/prediction" % URL, json={"dataset_path": "testdata.csv"}).content
response2 = requests.get("%s/scoring" % URL).content
response3 = requests.get("%s/summarystats" % URL).content
response4 = requests.get("%s/diagnostics" % URL).content


#combine all API responses
responses = response1 + "\n" + response2 + "\n" + response3 + "\n" + response4

#write the responses to your workspace

with open(os.path.join(model_path, "apireturns.txt"), "w") as returns_file:
    returns_file.write(responses)

