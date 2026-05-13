#!/bin/bash

# Check if WANDB_API_KEY is set
if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY environment variable is not set"
    echo "Set it with: export WANDB_API_KEY='your-key-here'"
    exit 1
fi

# Run the serving container with WandB API key from environment
docker run -p 5000:5000 \
  -e WANDB_API_KEY=${WANDB_API_KEY} \
  ift6758/serving:latest