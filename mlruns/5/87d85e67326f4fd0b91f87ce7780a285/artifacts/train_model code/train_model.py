from sklearn.linear_model import LinearRegression
import pickle
import pandas as pd
import os
 
import mlflow
from mlflow.tracking import MlflowClient
 
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("train_model_2")
 
df = pd.read_csv('/home/alexberkut98/datasets_2/data_train.csv', header=None)
df.columns = ['id', 'counts']
model = LinearRegression()
 
with mlflow.start_run():
    mlflow.sklearn.log_model(model,
                             artifact_path="lr",
                             registered_model_name="lr")
    mlflow.log_artifact(local_path="/home/alexberkut98/scripts_2/train_model.py",
                        artifact_path="train_model code")
    mlflow.end_run()
 
model.fit(df['id'].values.reshape(-1,1), df['counts'])
 
with open('/home/alexberkut98/models_2/data.pickle', 'wb') as f:
    pickle.dump(model, f)