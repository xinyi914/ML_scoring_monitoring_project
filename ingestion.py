import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    # print(input_folder_path)
    filenames = os.listdir(input_folder_path)
    df = pd.DataFrame(columns=["corporation","lastmonth_activity","lastyear_activity","number_of_employees","exited"])
    read_in_files = ""
    for file in filenames:
        path = os.path.join(os.getcwd(),input_folder_path,file)
        print(path)
        current_df = pd.read_csv(path)
        df = pd.concat([df,current_df],ignore_index=True).reset_index(drop=True)
    df = df.drop_duplicates()
    df.to_csv(os.path.join(os.getcwd(),output_folder_path,"finaldata.csv"))

    # create a record
 
    # filename = "finaldata.csv"
    output_location = "ingestedfiles.txt"

    # dataTimeobj = datetime.now()
    # theTimeNow = str(dataTimeobj.year)+'/'+str(dataTimeobj.month)+'/'+str(dataTimeobj.day)
    # all_records = [sourcelocation,filename,df.shape[0],theTimeNow]
    with open(os.path.join(os.getcwd(),output_folder_path,output_location),"w") as f:
        f.write(str(filenames))



if __name__ == '__main__':
    merge_multiple_dataframe()
