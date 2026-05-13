# HOW TO RUN:
# First ensure virtual environment/condas/ requirements.txt are all set up or installed. 
# 1. In Terminal #1:
#       cd serving folder
#       Then execute: python -m waitress --listen=0.0.0.0:5000 app:app
#
# 2. In Terminal #2:
#       cd client folder 
#       Then execute: python test_client.py

from serving_client import ServingClient
import pandas as pd

client = ServingClient(ip="127.0.0.1", port=5000)

print(" Predicting default ")
df = pd.DataFrame({"distance_from_net": [20, 45]})
print(client.predict(df))

print(" Switching the model ")
print(client.download_registry_model("IFT6758-2025-B08", "logreg_angle", "latest"))

print(" Predicting angle ")
df2 = pd.DataFrame({"angle_from_net": [10, 22]})
print(client.predict(df2))

print("Generating Logs ")
print(client.logs()["logs"])

