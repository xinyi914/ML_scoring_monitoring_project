from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import scoring
import diagnostics
# import predict_exited_from_saved_model
import json
import os



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['GET'])
def predict():        
    #call the prediction function you created in Step 3
    filename = request.args.get('filename')
    df = pd.read_csv(filename)
    X = df.drop(columns = ["corporation","exited"])
    preds = diagnostics.model_predictions(X)
    return 'preds: ' + str(preds) #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score(): 
    #check the score of the deployed model
    f1_score = scoring.score_model()
    return 'f1_score: ' + str(f1_score) #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    return 'stats: ' + '\n' + str(diagnostics.dataframe_summary()) #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnosis():        
    #check timing and percent NA values
    timing = diagnostics.execution_time()
    nas = diagnostics.check_missing()
    depend = diagnostics.outdated_packages_list()
    return 'timing: ' + str(timing) + '\n' + 'missing: ' + str(nas) + '\n' + 'dependicies: ' + '\n' + str(depend) #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
