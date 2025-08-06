
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

deploy_path = os.path.join(config['prod_deployment_path']) 
output_folder_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config['test_data_path']) 

##################Function to get model predictions
def model_predictions(df):
    #read the deployed model and a test dataset, calculate predictions
    with open(os.path.join(deploy_path,"trainedmodel.pkl"),"rb") as f:
        model = pickle.load(f)
    # test_data = pd.read_csv(os.path.join(test_data_path,"testdata.csv"))
    preds = list(model.predict(df))
    return preds #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    df = pd.read_csv(os.path.join(output_folder_path,"finaldata.csv"),index_col=0)
    means = df.mean()
    median = df.median()
    sd = df.std()
    return [means,median,sd]#return value should be a list containing all summary statistics

##################Function to get check missing data
def check_missing():
    df = pd.read_csv(os.path.join(output_folder_path,"finaldata.csv"),index_col=0)
    columns = df.columns
    nas = []
    for col in columns:
        nas.append(df[col].isna().sum()/df.shape[0])
    return nas

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    timing = []
    starttime = timeit.default_timer()
    os.system('python ingestion.py')
    timing.append(timeit.default_timer() - starttime)

    starttime = timeit.default_timer()
    os.system('python training.py')
    timing.append(timeit.default_timer() - starttime)


    return timing #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    df = pd.DataFrame(columns=["name","installed version","most recent version"])
    with open("requirements.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split("==")
            line[1] = line[1].strip()
            output = subprocess.check_output(['python','-m','pip','show',line[0]],text=True)
            for l in output.splitlines():
                if l.startswith("Version:"):
                    version = l.split(":",1)[1].strip()
            current_req = pd.DataFrame({"name": [line[0]], 
                                        "installed version": [line[1]],
                                        "most recent version": [version]})
            df = pd.concat([df,current_req],ignore_index = True)

    return df









if __name__ == '__main__':
    test_data = pd.read_csv(os.path.join(test_data_path,"testdata.csv"))
    X = test_data.drop(columns = ["corporation","exited"])
    preds = model_predictions(X)
    print("preds ",preds)
    print("------------------------------------------")
    stats = dataframe_summary()
    print("stats ",stats)
    print("------------------------------------------")
    missing = check_missing()
    print("missing ",missing)
    print("------------------------------------------")
    timing = execution_time()
    print("timing ",timing)
    print("------------------------------------------")
    df = outdated_packages_list()
    print("df")
    print(df)





    
