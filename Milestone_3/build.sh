#!/bin/bash

# Build the serving container
docker build -t ift6758/serving:latest -f Dockerfile.serving .

# Build the streamlit container
docker build -t ift6758/streamlit:latest -f Dockerfile.streamlit .

echo "Both images built successfully!"