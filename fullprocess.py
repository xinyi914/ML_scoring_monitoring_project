
import json
import os
import ast
import pickle
import pandas as pd
from sklearn import metrics
import subprocess
import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting
import apicalls


with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
deployment_path = config["prod_deployment_path"]

##################Check and read new data
#first, read ingestedfiles.txt
with open(os.path.join(deployment_path,"ingestedfiles.txt"),"r") as f:
    read_files = ast.literal_eval(f.read())

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
filenames = os.listdir(input_folder_path)
same_files = set(filenames) - set(read_files)

if same_files:
    # subprocess.run(['python','ingestion.py'])
    ingestion.merge_multiple_dataframe()
##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    with open(os.path.join(deployment_path,"latestscore.txt"),"r") as f:
        old_latest_f1_score = f.read()
    with open(os.path.join(deployment_path,"trainedmodel.pkl"),"rb") as f:
        model = pickle.load(f)
    new_data = pd.read_csv(os.path.join(output_folder_path,"finaldata.csv"),index_col=0)    
    X = new_data.drop(columns = ["corporation","exited"])
    y = new_data["exited"]
    preds_new = model.predict(X)
    new_f1_score = metrics.f1_score(y,preds_new)
    drift = new_f1_score < float(old_latest_f1_score)
    print("new f1 score: ", new_f1_score)
    print("old_latest f1 score: ",old_latest_f1_score)
##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
    if drift:
        # retraining
        training.train_model()
        scoring.score_model()
##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
        print("copying to deployment path")
        deployment.store_model_into_pickle()

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
        reporting.score_model()
        subprocess.run(['python','apicalls.py'])
    else:
        print("no drift")
else:
    print("same files")









