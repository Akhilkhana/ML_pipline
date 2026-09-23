import pandas as pd
import pickle
import yaml
import json
from sklearn.metrics import accuracy_score
import os
import mlflow

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Akhilkhana/machinelearningpipline.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "Akhilkhana"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "09b50ceb4d152fb348eaea8464410df91ff45c1f"
##if not os.getenv("MLFLOW_TRACKING_PASSWORD"):
   ## raise RuntimeError(
       ## "Set MLFLOW_TRACKING_PASSWORD to a DagsHub access token before evaluation."
   ## )

## load parameters from params.yaml
with open("params.yaml") as file:
    params = yaml.safe_load(file)["train"]

def evaluate(data_path,model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_registry_uri("https://dagshub.com/Akhilkhana/machinelearningpipline.mlflow")

    ## laod the model from the disk
    model= pickle.load(open(model_path, 'rb'))

    predictions = model.predict(X)
    accuracy= accuracy_score(y, predictions)
    mlflow.log_metric("accuracy", accuracy)
    with open("metrics.json", "w") as file:
        json.dump({"accuracy": float(accuracy)}, file, indent=2)
    print(f"Model accuracy: {accuracy}")

if __name__=="__main__":
    evaluate(params["data"],params["model"])     
