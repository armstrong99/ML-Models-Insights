from flask import Flask
from flask import request
import numpy as np
from joblib import load
#Load Azure Model and Workspace
from azureml.core import Workspace
from azureml.core.model import Model
ws = Workspace.from_config(path="Azure/Server/config/config.json")

# Create App using flask
app = Flask(__name__)
app.debug = True

# _app variables


# Loading ML model from Azure
def load_Azure_ml_model():
    global azure_model
    path = Model.get_model_path(model_name="customer_spent_lr", version=4, _workspace=ws)
    azure_model = load(path)    
    

# Prediction result from model
def prediction_Azure_ml_model(avg_session_length, time_on_app,  time_on_website, length_of_membership):
    try:
        print(int(avg_session_length))
        result = azure_model.predict([[
            int(avg_session_length), 
            int(time_on_website), 
            int(time_on_app), 
            int(length_of_membership)]])
        return result
    except Exception as e:
        error = str(e)
        print(error)
        return {"message": "An error occured in processing result", "data": error}



@app.route("/ml-predict-yamtst")
def handleIotInit():
    AvgSessionLength = request.args.get('AvgSessionLength')
    TimeonApp = request.args.get('TimeonApp')
    TimeonWebsite = request.args.get('TimeonWebsite')
    LengthofMembership = request.args.get('LengthofMembership')

    # Load Azure remote model
    load_Azure_ml_model()

    result = prediction_Azure_ml_model(AvgSessionLength, TimeonApp, TimeonWebsite, LengthofMembership)

    return '''<h1>The yearly amount spent is likely to be of value: {}</h1>'''.format(result[0])

if __name__ == '__main__':
    # run app in debug mode on port 9270
    app.run(debug=True, port=9270)

