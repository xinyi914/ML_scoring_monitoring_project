import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions
import seaborn as sns


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])



##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    test_data = pd.read_csv(os.path.join(test_data_path,"testdata.csv"))
    X = test_data.drop(columns = ["corporation","exited"])
    y = test_data["exited"].values.reshape([-1,1])
    preds = model_predictions(X)
    # print(y)
    # print(preds)
    cm = metrics.confusion_matrix(y,preds)

    # plot and save
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
    plt.xlabel('predicted')
    plt.ylabel("actual")
    plt.savefig(os.path.join(output_model_path,"confusionmatrix.png"))


if __name__ == '__main__':
    score_model()
