import mlflow
import dagshub
import json
from pathlib import Path
from mlflow import MlflowClient
import logging
import os

# create logger
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# create a fomratter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to handler
handler.setFormatter(formatter)

# initialize dagshub
import dagshub
import mlflow.client
import dagshub
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"  
repo_owner = "Sovith07"
repo_name = "swiggy_delivery_time"

mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info


if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    
    # run information file path
    run_info_path = root_path / "run_information.json"
    
    # register the model
    run_info = load_model_information(run_info_path)
    
    # get the run id
   
    
    # model to register path
    model_name = "delivery_time_pred_model"
    model_uri = run_info['model_uri']
    
    # register the model
    model_version=mlflow.register_model(model_uri, model_name)
    
    
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
    
    
    
    logger.info("Model pushed to Staging stage")

