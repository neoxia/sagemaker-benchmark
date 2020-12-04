#!/bin/bash

set -e

echo "Executing: python3 projects/$1/pipeline/main.py --mode "$2""

# installing dependencies
pip install boto3
pip install PyYAML
pip install sagemaker==2.15.0

# running the command
# mode = train or deploy
# ex: python3 project/mnist_tensorflow2_scriptmode/pipeline/main.py --mode train
python3 "projects/$1/pipeline/main.py" --mode "$2"