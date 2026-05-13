import json
import requests
import pandas as pd
import logging


logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5000, features=None):
        """
        Simple client wrapper for interacting with the Flask prediction service.
        """
        # Defining self.base_url
        self.base_url = f"http://{ip}:{port}"
        # Add information to logger.info
        logger.info(f"Initializing client; base URL: {self.base_url}")
        # If features is none, automatically assign features to distance_from_net 
        if features is None:
            features = ["distance_from_net"]
        self.features = features

        # any other potential initialization

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """
        # If instance is not a pd.DataFrame then raise a ValueError 
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X is mandated to be a pandas Dataframe.")
        # Convert the DataFrame into a dictionary of lists which will match the Flask input
        payload_f = X.to_dict(orient="list")
        url = f"{self.base_url}/predict"
        logger.info(f"Sending the prediction request to the destination: {url}")

        # Try to to see if you could put requests.post into response 
        try:
            response = requests.post(url, json=payload_f)
        # Catching Exception as e
        except Exception as e:
            # Logging the error 
            logger.error(f"Failed to obtain the proper prediction endpoint: {e}")
            raise 
        # If response status code does not equal 200 then prediction fails
        if response.status_code !=200:
            logger.error(f"The Prediction has failed because: {response.text}")
            raise RuntimeError(f"Prediction request has failed: {response.text}")
        # Data which will be used for predictions 
        data_preds = response.json()

        ## If Predictions not in data raise a Runtime Error
        if "predictions" not in data_preds:
            raise RuntimeError(f"Unexpected  response structure for predictions: {data_preds}")
        # Defining predictions
        predictions = data_preds["predictions"]

        # Return results as a Dataframe which is aligned with the given input index 
        return pd.DataFrame({"prediction": predictions}, index=X.index)
    
    def logs(self) -> dict:
        """Get server logs"""
        # Defining url and logger.info
        url = f"{self.base_url}/logs"
        logger.info(f"We are requesting logs from {url}")

        # Try and except 
        try:
            # Define response 
            response = requests.get(url)
        except Exception as e:
            # Define logger.error 
            logger.error(f"Failed to achieve the logs endpoint {e}")
            raise 
        # If response.status_code does not equal 200
        if response.status_code != 200:
            logger.error(f"Log request has failed because of: {response.text}")
            raise RuntimeError(f"Logs request has failed because of: {response.text}")
        
        # Returning response.json
        return response.json()
    
    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """

        # Defining url 
        url = f"{self.base_url}/download_registry_model"

        # Defining payload 
        payload = {
            "model": model, 
            "version": version
        }
        # Logging info
        logger.info(f"Sending download_registry_model request to {url} with this {payload}")

        # try and except 
        try:
            response = requests.post(url, json=payload)
        except Exception as e:
            logger.error(f"Failed to properly get to model download endpoint {e}")
            raise 
        # if response.status_code != 200, log the error 
        if response.status_code != 200:
            logger.error(f"Model download has failed: {response.text}")
            raise RuntimeError(f"Model download has failed: {response.text}")
        
        # return response.json
        return response.json()

