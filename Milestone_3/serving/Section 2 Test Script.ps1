## Within your conda, environment, or VS code, install the requirements file in the VS code terminal

## For each of our runs, we created a virtual environment with the code as follows in your VS Code terminal:

cd "main folder" in my version it is "C:\Users\Dell\OneDrive\Desktop\UdeM Graduate\Data Science\Milestone3"

## Then run:

python -m venv serving_env

## Then run:

serving_env\Scripts\activate

## Then run:

pip install --upgrade pip

## in the same terminal with the following command
pip install -r requirements.txt


## Note that Online Link where you can view the log
Logs Link: 127.0.0.1:5000/logs

Example Test Script:

## Before running test for flask app, make sure you have severed the connection to any
## older flask runs, and have deleted any pycaches or models which have been used before 

## Note this presumes you run the test on windows, UBUNTU will need CURL commands 

# First access the serving folder with the commands cd serving 
# In the first VS Code Terminal, run line below 
python -m waitress --listen=0.0.0.0:5000 app:app

# Create a new terminal and execute the following commands 
# Test logs
# Outputs message indicating successful loading of default model logreg_distance
# First few commands may fail due to server not being fully connected yet
# However, API call should execute properly after 2 to 3 minutes in the worst-case scenario
Invoke-WebRequest -Uri "http://127.0.0.1:5000/logs" | Select-Object -ExpandProperty Content

# Test default distance model, we should expect a probability value returned 
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST -ContentType "application/json" -Body '{"distance_from_net":[20]}'

# Wrong field test, we should expect an error which indicates incorrect field inputted. 
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST -ContentType "application/json" -Body '{"angle_from_net":[30]}'

# Switch to angle model, we should expect a successful switch to new model message  
Invoke-RestMethod -Uri "http://127.0.0.1:5000/download_registry_model" -Method POST -ContentType "application/json" -Body '{"model":"logreg_angle","version":"latest"}'

# We should expect a probability value returned here 
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST -ContentType "application/json" -Body '{"angle_from_net":[30]}'

# Switch to distance+angle model, we should see a successful message for model switch returned
Invoke-RestMethod -Uri "http://127.0.0.1:5000/download_registry_model" -Method POST -ContentType "application/json" -Body '{"model":"logreg_distance_angle","version":"latest"}'

# Predicting goal based on distance and angle combined, we should expect a probability output here
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST -ContentType "application/json" -Body '{"distance_from_net":[20], "angle_from_net":[30]}'

# Missing JSON body, should expect an error message
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST

# Wrong format (string instead of dict/list), should expect an error message 
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST `
-ContentType "application/json" -Body '"hello"'

# Empty JSON Body, should expect an error message
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST `
-ContentType "application/json" -Body '{}'

# Empty model to download, should expect an error message
Invoke-RestMethod -Uri "http://127.0.0.1:5000/download_registry_model" -Method POST `
-ContentType "application/json" `
-Body '{"model":"not_a_real_model", "version":"v999"}'

# Call successfully extracts latest version of logreg_distance, should expect successful message 
Invoke-RestMethod -Uri "http://127.0.0.1:5000/download_registry_model" -Method POST `
-ContentType "application/json" `
-Body '{"model":"logreg_distance", "version":"latest"}'

# Should see log message 
Invoke-WebRequest -Uri "http://127.0.0.1:5000/logs" | Select-Object -ExpandProperty Content

# Expect to see 2 successful predictions derived from distance 
## we drop angle_from_net feature quietly as they are unnecessary for the input 
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" `
-Method POST -ContentType "application/json" `
-Body '{"distance_from_net":[20, 35], "angle_from_net":[30, 40]}'
