import requests
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:"



#Call each API endpoint and store the responses
response1 = requests.get(URL+"8000/prediction?filename=testdata/testdata.csv").content.decode('utf-8')#put an API call here
response2 = requests.get(URL+"8000/scoring").content.decode('utf-8')#put an API call here
response3 = requests.get(URL+"8000/summarystats").content.decode('utf-8')#put an API call here
response4 = requests.get(URL+"8000/diagnostics").content.decode('utf-8')#put an API call here

#combine all API responses
responses = response1 + '\n' + response2 + '\n'+ response3 + '\n' + response4 + '\n'#combine reponses here

#write the responses to your workspace
with open('config.json','r') as f:
    config = json.load(f) 

output_model_path = config['output_model_path']

with open(os.path.join(output_model_path,"apireturns.txt"),"w") as f:
    f.write(responses)



