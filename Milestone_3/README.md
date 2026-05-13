# Milestone 3 Repo Overview 

This readme explains how to implement Milestone3 in terms of implementing the flask app, docker files, and streamlit_app to generate quick predictions of our NHL data produced from Milestone 1 and Milestone 2. The immediate section below explains the repository folder structure. 

Note the WANDB API KEY you can use for this milestone is: 
**50c6486ca894b323e5061c4513b477036ac87f8a**

Note, the project was run with Python version **3.13.3**, so you likely need this version to ensure that the **.py** files run smoothly on your laptop if you want to run them without the **Docker**. 

After running the docker commands, the terminal will print the link: `http://0.0.0.0:8501`, **which will not work**. Please use the following link: `http://localhost:8501`.

## Repo Structure 

### figures 
Contains supporting images and data used in section 5 and 7.  

### ift6758/client/

Contains all files for **Section 3**: Game Client & Serving Client.

game_client.py — Retrieves NHL game data and preprocesses events.

test_client.py — Unit tests for the game client.

serving_client.py — Client used to send requests to the Flask model server.

test_sclient.py — Tests for the serving client.

This directory supports data retrieval and service interaction for Section 3.

### images/

Contains static assets (e.g., rink diagrams, graphics) used by notebooks and Streamlit.

### notebooks/

Includes exploratory notebooks used during analysis and debugging.

### serving/

Contains all files related to **Section 2**: Model Serving.

app.py — Main Flask application that loads models and exposes prediction endpoints.

Section 2 Test Script.ps1 — PowerShell script used to test the Flask API locally.

These files implement and validate the serving pipeline for Section 2.

###  Docker & Deployment Files

Used for Section 4 and Section 6 (containerization and deployment).

Dockerfile.serving — Builds the Docker image for the Flask inference server.

Dockerfile.streamlit — Builds the Docker image for the Streamlit app.

docker-compose.yaml — Orchestrates multi-service deployments (Flask + Streamlit).

build.sh — Script to build Docker images.

### streamlit_app.py

Implements the UI for Section 5 and Section 7.

Launches the Streamlit interface used for visualization and interactive predictions.

### Other Key Files

requirements.txt — Lists all Python dependencies required for the environment.

.gitignore — Specifies files and folders excluded from version control.

run.sh — Helper script for executing parts of the project.

## Installation

**Note: this is slightly different than the original rep you were working in!**

To install this package, first setup your Python environment (next section), and then simply run

    pip install -e ift6758

assuming you are running this from the root directory of this repo. 

## Docker

Follow the instructions [here](https://docs.docker.com/get-docker/) to install Docker on your system.
Included in this repo is a Dockerfile with brief explanations of the commands you will need to use
to use to define your Docker image.
One thing to note is that its generally a better idea to stick to using `pip` instead of Conda
environments in Docker containers, as this approach is generally more lightweight than Conda.
A common pattern is to copy the `requirements.txt` file into the docker container and then simply 
do `pip install -r requirements.txt` (done in the Dockerfile).
You can also copy an entire Python package (e.g. the `ift6758/` folder) into the container, and also
do a simple `pip install -e ift6758/` (also from the Dockerfile).

In addition, below are a list of useful docker commands that you will likely need.

### docker build ([ref](https://docs.docker.com/engine/reference/commandline/build/))
Builds a docker image directly the local Dockerfile (in the same directory):

```bash
# docker build -t <TAG>:<VERSION> .
# eg: 
docker build -t ift6758/serving:1.0.0 .
```

### docker images ([ref](https://docs.docker.com/engine/reference/commandline/images/))

Lists all images that are built.


### docker run ([ref](https://docs.docker.com/engine/reference/commandline/run/))
To run the docker image you just created, you could run:

```bash
#  docker run [OPTIONS] IMAGE [COMMAND] [ARG...]
docker run -it --expose 127.0.0.1:8890:8890/tcp --env DOCKER_ENV_VAR=$LOCAL_ENV_VAR ift6758/serving:0.0.1 
```

In this example, `-it --expose 127.0.0.1:8890:8890/tcp --env DOCKER_ENV_VAR=$LOCAL_ENV_VAR` are the
`[OPTIONS]`, `ift6758/serving:0.0.1` is the `IMAGE`, and there are no `[COMMAND]` or `[ARG...]`.
If you run this command the docker container will run whatever you specified at the `CMD` in your 
Dockerfile; if this is not specified or it crashes, your container will immediately stop.
You could alternatively specify a different command; for example setting `[COMMAND]` to `bash` will
drop you into a bash shell in the container. 
From there you can poke around to potentially debug your app.

Some useful run options are:

- `-it`: Allocate a pseudo TTY connected to the container's STDIN (i.e. interactive mode)
- `-p/--expose`: Expose port (e.g. `-p 127.0.0.1:80:8080/tcp` binds port `8080` of the container TCP port `80` on `127.0.0.1` of the host machine.
- `-e/--env`: Set environment variable (e.g. `-e DOCKER_ENV_VAR=${LOCAL_ENV_VAR}`)
- `-d/--detach`: runs container in the background

The documentation for all of these can be found in the [official Docker docs](https://docs.docker.com/engine/reference/commandline/run)

### docker ps ([ref](https://docs.docker.com/engine/reference/commandline/ps/))

Lists running containers.
For example, if you did a `docker run` command in detached mode (with the `-d` flag), your container
will be running in the background.
To verify that your container is running, you could run `docker ps`: 

```bash
> docker ps
CONTAINER ID   IMAGE                   COMMAND   CREATED         STATUS         PORTS                      NAMES
0237661ace81   ift6758/serving:0.0.1   "bash"    4 seconds ago   Up 3 seconds   127.0.0.1:8890->8890/tcp   sleepy_jang
```

### docker exec ([ref](https://docs.docker.com/engine/reference/commandline/exec/))

This runs a command in the container.
You could use this to drop into a shell in a detached but running container, eg:

```bash
> docker exec -it sleepy_jang bash
root@0237661ace81:/code# 
```

### docker network ([ref](https://docs.docker.com/engine/reference/commandline/network/))

Allows you to ping your docker network. You can do:

```bash
> docker network ls
NETWORK ID     NAME                 DRIVER    SCOPE
15742e644eb4   bridge               bridge    local
60c57e381d21   host                 host      local
```

to see all of your existing networks.
If you didn't build with docker compose, chances are your running containers are living on the
`bridge` network.
You can then do:

```bash
> docker network inspect bridge
...
"Containers": {                                                                                                                                                                                             
            "<...some id...>": {                                                                                                                                   
                "Name": "sleepy_jang",                                                                                                                                                                          
                "EndpointID": "...",
                "MacAddress": "...",
                "IPv4Address": "172.17.0.1",  #  <--- this is the ip of the container on the docker network!
                "IPv6Address": ""
            }
...
```

or any other network you may want to inspect to get more information about the containers attached.
For example, you can find the IP of a container by the NAMES.
This is the IP of the container on the **docker network**; i.e. if you were trying to make an
HTTP request from *within* the docker network, rather than from your local host.
This may be useful to you for debugging the final part of Milestone 3, where you will put
your jupyter notebook into a container and then query the prediction service that lives in another
docker container.

**Note when using docker compose**

Docker compose does some nice name resolution stuff for you by default.
You'll notice the format of a `docker-compose.yaml` file is along the lines of:

```yaml
services:
    service1:
        ...
    service2:
        ...
```

Say your jupyter notebook lives in `service2`, and you want to make an HTTP request to `service1`.
You actually don't need to look for the container IP of `service1` - you can simply make an 
HTTP request to `http://service1:PORT/endpoint`.
The name resolution is taken care for you if you're using docker compose (but you do need to keep
track of the port).

### docker-compose ([ref](https://docs.docker.com/compose/))

_The following is taken directly from the docker compose reference_

Compose is a tool for defining and running multi-container Docker applications. 
With Compose, you use a YAML file to configure your application’s services. 
Then, with a single command, you create and start all the services from your configuration.

Using Compose is basically a three-step process:

- Define your app’s environment with a Dockerfile so it can be reproduced anywhere.
- Define the services that make up your app in docker-compose.yml so they can be run together in an isolated environment.
- Run `docker-compose up` and the Docker compose command starts and runs your entire app.

Install it with:

```bash
pip install docker-compose
```

You can then simply do `docker-compose up` to build and run the application that you sepcified in
the `docker-compose.yaml` file.

## Generating New API (Section 1)

Generating this new API was already completed in Milestone 1 and 2. It is implicitly used in the Flask app, streamlit, and docker. 

## Flask App (Section 2)

To run the app, if you are in the same directory as `app.py`, you can run the app from the command 
line using gunicorn (Unix) or waitress (Unix + Windows):
    
```bash
gunicorn --bind 0.0.0.0:<PORT> app:app

# or

waitress-serve --listen=0.0.0.0:<PORT> app:app
```
Gunicorn or waitress can be installed via:

```bash
pip install gunicorn

# or

pip install waitress
```

To run the specific app.py (Flask app code) look at the detailed instructions in the Section 2 Test Script.ps1. The first few instructions are written out here to allow you to begin your flask app testing. 

Instructions: 

Within your conda, environment, or VS code, install the requirements file in the VS code terminal
in the same terminal with the following command

For each of our runs, we created a virtual environment with the code as follows in your VS Code terminal: 
```
cd "main folder" in my version it is "C:\Users\Dell\OneDrive\Desktop\UdeM Graduate\Data Science\Milestone3"
```
Then run:
```
python -m venv serving_env
```

Then run: 
```
serving_env\Scripts\activate
```

Then run:

```
pip install -r requirements.txt
```

Note that Online Link where you can view the log
Logs Link: 127.0.0.1:5000/logs

Example Test Script:

Before running test for flask app, make sure you have severed the connection to any
older flask runs, and have deleted any pycaches or models which have been used before 

Note this presumes you run the test on windows, UBUNTU will need CURL commands 

First access the serving folder with the commands cd serving 

In the first VS Code Terminal, run line below 
```
python -m waitress --listen=0.0.0.0:5000 app:app
```

Create a new terminal and execute the following commands 
Test logs
Outputs message indicating successful loading of default model logreg_distance
First few commands may fail due to server not being fully connected yet
However, API call should execute properly after 2 to 3 minutes in the worst-case scenario
```
Invoke-WebRequest -Uri "http://127.0.0.1:5000/logs" | Select-Object -ExpandProperty Content
```

Test default distance model, we should expect a probability value returned 
```
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST -ContentType "application/json" -Body '{"distance_from_net":[20]}'
```

Wrong field test, we should expect an error which indicates incorrect field inputted. 
```
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST -ContentType "application/json" -Body '{"angle_from_net":[30]}'
```

Switch to angle model, we should expect a successful switch to new model message  
```
Invoke-RestMethod -Uri "http://127.0.0.1:5000/download_registry_model" -Method POST -ContentType "application/json" -Body '{"model":"logreg_angle","version":"latest"}'
```

We should expect a probability value returned here 
```
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST -ContentType "application/json" -Body '{"angle_from_net":[30]}'
```

For the rest of the testing, please complete the rest of the commands in Section 2 Test Script.ps1. Note you can also modify the testing commands as you see fit to ensure everything is working. Once you are finished with the testing, you can press control +c in the vs code which is running the flask app. 

## Testing Serving and Game Client (Section 3)

To test the **serving_client file**, follow these instructions below:

HOW TO RUN:

First ensure your virtual environments and requirements are downloaded with the instruction specified in section 2 (Flask app).

Then run these commands: 

1. In VS Code, within your first Terminal, access the serving folder with:
   ```
     cd serving or (correct serving path for your downloaded folder)
   ```
2. Then execute the flask initialization command below:
   ```
   python -m waitress --listen=0.0.0.0:5000 app:app
   ```  
3. Open a second VS code terminal, and in your second VS code Terminal, access the client folder with:
   ```
   cd client (or given client path folder)
   ```
4. Then
   ```
   execute: python test_client.py
   ```
   You can modify the test commands as you see fit to ensure the code is functioning smoothly.

To test the **game_client** file, follow these instructions below:

1. Open a second VS code terminal, and in your second VS code Terminal, access the client folder with:
   ```
   cd client (or given client path folder)
   ```
5. Then
   ```
   execute: python test_gclient.py
   ```
   You can modify the test commands as you see fit to ensure the code is functioning smoothly.

   Once you are finished with the testing, you can press control +c in the vs code which is running the flask app. 
   
## Streamlit (Sections 5 & 7)

After entering the workspace, model name, model version, the model can be downloaded from WandB. When the game ID, e.g., 2021020329, is given and the `Ping game` button is pressed, it would be possible to select the `event ID`, after which the match data will be displayed. 

This consists of the expected goals along with some other statistics, the predictions and the data used upto the chosen event. The shot heatmaps of both teams for the entire match are also shown.

The application has been structured in a robust manner and can handle various cases and orders of inputting the data.

## Docker Deployment (Tasks 4 & 6 - Milestone 3)

This project includes a complete Dockerized deployment system with two containerized services:
- **Flask Serving Service** (Port 5000): Model serving API with hot-swapping capabilities
- **Streamlit Dashboard** (Port 8501): Interactive web interface for predictions

### Prerequisites

1. **Install Docker Desktop**: Follow instructions at https://docs.docker.com/get-docker/
2. **WandB API Key**: You will need your WandB API key to download models

### Quick Start Guide

#### Step 1: Set WandB API Key

**Before running Docker**, set your WandB API key in the terminal:

**Windows PowerShell:**
```powershell
$env:WANDB_API_KEY="your-wandb-api-key-here"
```

**Linux/Mac/Git Bash:**
```bash
export WANDB_API_KEY="your-wandb-api-key-here"
```

**Verify it's set:**
```powershell
# Windows
echo $env:WANDB_API_KEY

# Linux/Mac
echo $WANDB_API_KEY
```

#### Step 2: Build and Run

From the project root directory, run:
```bash
docker-compose up --build
```

This single command will:
- Build both Docker images (serving and streamlit)
- Start the Flask serving container on port 5000
- Start the Streamlit dashboard on port 8501
- Configure networking between containers

**To run in background (detached mode):**
```bash
docker-compose up --build -d
```

#### Step 3: Access Services

Once containers are running:
- **Flask API Logs**: http://localhost:5000/logs
- **Streamlit Dashboard**: http://localhost:8501

**Important**: Use `localhost` NOT `0.0.0.0` in your browser!

### Docker Services Overview

#### Flask Serving Service (Port 5000)

The serving service provides three REST API endpoints:

**1. `/predict` (POST)** - Make goal probability predictions
```powershell
# Windows PowerShell
$testData = @{distance_from_net = @(50, 60, 70)} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -Body $testData -ContentType "application/json"
```
```bash
# Linux/Mac
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"distance_from_net": [50, 60, 70]}'
```

**2. `/logs` (GET)** - View application logs
```powershell
# Windows
Invoke-WebRequest -Uri http://localhost:5000/logs

# Linux/Mac
curl http://localhost:5000/logs
```

**3. `/download_registry_model` (POST)** - Download and hot-swap models
```powershell
# Windows
$model = @{model = "logreg_distance_angle"; version = "v0"} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:5000/download_registry_model -Method Post -Body $model -ContentType "application/json"

# Linux/Mac
curl -X POST http://localhost:5000/download_registry_model \
  -H "Content-Type: application/json" \
  -d '{"model": "logreg_distance_angle", "version": "v0"}'
```

#### Streamlit Dashboard (Port 8501)

Interactive web dashboard providing:
- Model selection and downloading from WandB
- Game ID input for NHL games
- Real-time predictions display
- Expected goals (xG) calculations
- Shot events data visualization

Access at: http://localhost:8501

### Alternative: Manual Build and Run

#### Build Images Using Scripts

**Build both images:**
```bash
# Make script executable (Linux/Mac)
chmod +x build.sh

# Run build script
./build.sh
```

**Or build manually:**
```bash
# Build serving image
docker build -t ift6758/serving:latest -f Dockerfile.serving .

# Build streamlit image
docker build -t ift6758/streamlit:latest -f Dockerfile.streamlit .
```

#### Run Individual Containers

**Important**: Set WANDB_API_KEY first!

**Run serving container:**
```bash
# Make script executable (Linux/Mac)
chmod +x run.sh

# Run script
./run.sh
```

**Or run manually:**
```bash
# Windows PowerShell
docker run -p 5000:5000 -e WANDB_API_KEY=$env:WANDB_API_KEY ift6758/serving:latest

# Linux/Mac
docker run -p 5000:5000 -e WANDB_API_KEY=${WANDB_API_KEY} ift6758/serving:latest
```

**Run streamlit container:**
```bash
# Windows PowerShell
docker run -p 8501:8501 -e WANDB_API_KEY=$env:WANDB_API_KEY ift6758/streamlit:latest

# Linux/Mac
docker run -p 8501:8501 -e WANDB_API_KEY=${WANDB_API_KEY} ift6758/streamlit:latest
```

**Note**: When running containers individually, they cannot communicate with each other. Use `docker-compose` for full functionality.

### Docker Commands Reference

#### View Running Containers
```bash
docker ps
```

Expected output:
```
CONTAINER ID   IMAGE                      PORTS                    NAMES
abc123...      ift6758/serving:latest     0.0.0.0:5000->5000/tcp   ift6758-serving
def456...      ift6758/streamlit:latest   0.0.0.0:8501->8501/tcp   ift6758-streamlit
```

#### View Container Logs
```bash
# View serving logs
docker logs ift6758-serving

# View streamlit logs
docker logs ift6758-streamlit

# Follow logs in real-time
docker logs -f ift6758-serving
```

#### Stop Containers
```bash
# Stop all services
docker-compose down

# Stop specific service
docker-compose stop serving
```

#### Restart Services
```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart serving
```

#### Rebuild After Code Changes
```bash
# Rebuild and restart
docker-compose up --build

# Rebuild without cache (clean build)
docker-compose build --no-cache
docker-compose up
```

### Testing the Docker Deployment

#### Test 1: Flask Endpoints

**Test logs endpoint:**
```powershell
# Windows
Invoke-WebRequest -Uri http://localhost:5000/logs
```
```bash
# Linux/Mac
curl http://localhost:5000/logs
```

Should show successful model loading in logs.

**Test prediction endpoint:**
```powershell
# Windows
$testData = @{distance_from_net = @(50, 60, 70)} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -Body $testData -ContentType "application/json"
```
```bash
# Linux/Mac
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"distance_from_net": [50, 60, 70]}'
```

Should return prediction probabilities.

**Test model download:**
```powershell
# Windows
$modelRequest = @{model = "logreg_angle"; version = "latest"} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:5000/download_registry_model -Method Post -Body $modelRequest -ContentType "application/json"
```
```bash
# Linux/Mac
curl -X POST http://localhost:5000/download_registry_model \
  -H "Content-Type: application/json" \
  -d '{"model": "logreg_angle", "version": "latest"}'
```

Should confirm successful model download.

#### Test 2: Client Integration

Ensure Docker containers are running:
```bash
docker-compose up -d
```

Navigate to client directory and run tests:
```bash
cd ift6758/ift6758/client
python test_client.py
```

Expected output should show successful predictions and model swapping.

#### Test 3: Streamlit Dashboard

1. Open browser to: http://localhost:8501
2. In sidebar, enter model details:
   - Workspace: `IFT6758-2025-B08`
   - Model Name: `logreg_distance`
   - Version: `v8`
3. Click "Download Model" - should see success message
4. Enter Game ID: `2021020329`
5. Click "Ping Game" - should display game data and predictions

**Key Features:**
- Containers communicate via Docker network using service names
- Streamlit uses `http://serving:5000` to connect to Flask
- Host machine accesses via `localhost:5000` and `localhost:8501`

### Environment Variables

| Variable | Required | Description | How to Set |
|----------|----------|-------------|------------|
| `WANDB_API_KEY` | Yes | WandB API key for model registry | Set in terminal before running docker-compose |

**Setting the API key:**
```powershell
# Windows PowerShell (valid for current session)
$env:WANDB_API_KEY="your-key-here"

# Linux/Mac (valid for current session)
export WANDB_API_KEY="your-key-here"
```

**Note**: The API key must be set in the terminal each time you open a new session.

### Troubleshooting

#### Issue: "WANDB_API_KEY variable is not set"

**Cause**: Environment variable not set before running docker-compose.

**Solution**: 
```powershell
# Windows
$env:WANDB_API_KEY="your-wandb-api-key"

# Linux/Mac
export WANDB_API_KEY="your-wandb-api-key"

# Then run
docker-compose up
```

#### Issue: "Port 5000 already in use"

**Cause**: Another application is using port 5000.

**Solution 1**: Stop the conflicting application
```powershell
# Windows - Find process using port
netstat -ano | findstr :5000
# Kill the process (replace PID with actual number)
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :5000
kill -9 <PID>
```

**Solution 2**: Change port in `docker-compose.yaml`
```yaml
ports:
  - "5001:5000"  # Use port 5001 instead
```

#### Issue: "Cannot reach http://0.0.0.0:8501/"

**Cause**: Using wrong URL in browser.

**Solution**: Use `http://localhost:8501` NOT `http://0.0.0.0:8501`

#### Issue: "Streamlit cannot connect to serving"

**Cause**: Incorrect IP address in `streamlit_app.py`.

**Solution**: Ensure `streamlit_app.py` uses Docker service name:
```python
# Correct - uses Docker service name
client = ServingClient(ip="serving", port=5000)

# Wrong - will not work inside Docker
client = ServingClient(ip="localhost", port=5000)
```

#### Issue: Container crashes immediately

**Cause**: Various (syntax errors, missing files, import errors).

**Solution**: Check container logs:
```bash
docker logs ift6758-serving
docker logs ift6758-streamlit
```

Look for error messages and fix accordingly.

#### Issue: "Model failed to load"

**Possible causes:**
1. WANDB_API_KEY not set or incorrect
2. Model doesn't exist in WandB registry
3. Network connection issues

**Solution**: 
1. Verify API key is correct
2. Check model exists in WandB workspace
3. View detailed logs: `docker logs ift6758-serving`

#### Issue: "NumPy version mismatch" (local testing only)

**Cause**: Local Python has NumPy 2.x but pandas requires 1.x.

**Solution**: This only affects local testing, not Docker. Fix with:
```bash
pip install "numpy<2"
```

### Clean Up Docker Resources

**Remove stopped containers:**
```bash
docker-compose down
```

**Remove images:**
```bash
docker rmi ift6758/serving:latest
docker rmi ift6758/streamlit:latest
```

**Remove all unused Docker resources:**
```bash
docker system prune -a
```

**Warning**: This removes all stopped containers, unused networks, dangling images, and build cache.

### Development Workflow

1. **Make code changes** to `serving/app.py` or `streamlit_app.py`
2. **Rebuild containers**: 
```bash
   docker-compose down
   docker-compose up --build
```
3. **Test changes**: 
   - Flask: http://localhost:5000/logs
   - Streamlit: http://localhost:8501
4. **Check logs for errors**: 
```bash
   docker logs -f ift6758-serving
   docker logs -f ift6758-streamlit
```
5. **Iterate**: Repeat steps 1-4 as needed

### Complete Test Workflow

**Step-by-step testing procedure:**
```bash
# 1. Set API key
export WANDB_API_KEY="your-key"  # or $env:WANDB_API_KEY on Windows

# 2. Clean previous builds
docker-compose down
docker system prune -f

# 3. Build and start
docker-compose up --build

# 4. Wait 30 seconds for startup
# (Open new terminal for testing)

# 5. Test Flask logs
curl http://localhost:5000/logs

# 6. Test Flask prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"distance_from_net": [50]}'

# 7. Test Streamlit
# Open browser to http://localhost:8501

# 8. Test client integration
cd ift6758/ift6758/client
python test_client.py

# 9. Check everything passed
# If all tests pass, deployment is successful!
```

### Additional Resources

- [Docker Official Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [WandB Python SDK](https://docs.wandb.ai/ref/python)

### Notes

- Docker containers are isolated environments - changes to local files won't affect running containers until you rebuild
- Always set WANDB_API_KEY before running docker-compose
- Use `localhost` (not `0.0.0.0`) when accessing services from your browser
- Inside Docker containers, services communicate using service names (e.g., `http://serving:5000`)
- Check logs regularly for debugging: `docker logs -f <container-name>`


## Environments

The first thing you should setup is your isolated Python environment.
You can manage your environments through either Conda or pip.
Both ways are valid, just make sure you understand the method you choose for your system.
It's best if everyone on your team agrees on the same method, or you will have to maintain both environment files!
Instructions are provided for both methods.

**Note**: If you are having trouble rendering interactive plotly figures and you're using the pip + virtualenv method, try using Conda instead.

### Conda 

**Note: it is better to stick with pip environments in Docker containers!**

Conda uses the provided `environment.yml` file.
You can ignore `requirements.txt` if you choose this method.
Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual) installed on your system.
Once installed, open up your terminal (or Anaconda prompt if you're on Windows).
Install the environment from the specified environment file:

    conda env create --file environment.yml
    conda activate ift6758-conda-env

After you install, register the environment so jupyter can see it:

    python -m ipykernel install --user --name=ift6758-conda-env

You should now be able to launch jupyter and see your conda environment:

    jupyter-lab

If you make updates to your conda `environment.yml`, you can use the update command to update your existing environment rather than creating a new one:

    conda env update --file environment.yml    

You can create a new environment file using the `create` command:

    conda env export > environment.yml

### Pip + Virtualenv

An alternative to Conda is to use pip and virtualenv to manage your environments.
This may play less nicely with Windows, but works fine on Unix devices.
This method makes use of the `requirements.txt` file; you can disregard the `environment.yml` file if you choose this method.

Ensure you have installed the [virtualenv tool](https://virtualenv.pypa.io/en/latest/installation.html) on your system.
Once installed, create a new virtual environment:

    vitualenv ~/ift6758-venv
    source ~/ift6758-venv/bin/activate

Install the packages from a requirements.txt file:

    pip install -r requirements.txt

As before, register the environment so jupyter can see it:

    python -m ipykernel install --user --name=ift6758-venv

You should now be able to launch jupyter and see your conda environment:

    jupyter-lab

If you want to create a new `requirements.txt` file, you can use `pip freeze`:

    pip freeze > requirements.txt



