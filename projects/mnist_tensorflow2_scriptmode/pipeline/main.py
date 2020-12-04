
# import libraries
import argparse
import os
import json
import datetime
import boto3
import sagemaker
from sagemaker.tensorflow.estimator import TensorFlow
from sagemaker.tensorflow.model import TensorFlowModel


from env import *


def train():
    "Train the model in SageMaker"
    
    with open(HYPERPARAMETERS_FILE_PATH) as f:
        hyperparameters = json.load(f)
    
    # define the estimator
    print("Build Estimator...")
    estimator = TensorFlow(entry_point=ENTRY_POINT_TRAIN,
                           source_dir=SOURCE_DIR,
                           output_path=MODEL_ARTIFACTS_S3_LOCATION,
                           code_location=CUSTOM_CODE_TRAIN_UPLOAD_S3_LOCATION,
                           base_job_name=BASE_JOB_NAME,
                           role=ROLE_ARN,
                           py_version=PYTHON_VERSION,
                           framework_version=FRAMEWORK_VERSION,
                           hyperparameters=hyperparameters,
                           instance_count=TRAIN_INSTANCE_COUNT,
                           instance_type=TRAIN_INSTANCE_TYPE,
                           distributions=DISTRIBUTIONS)
    
    # train the model
    print("Fit the estimator...")
    estimator.fit({"train": TRAIN_DATA_S3_LOCATION, "test": TEST_DATA_S3_LOCATION})
    
    print("Store the training job name...")
    with open(FILENAME_TRAINING_JOB_NAME, "w+") as f:
        f.write(str(estimator.latest_training_job.name))
    
    return estimator

    
def deploy():
    "Deploy the model in a SageMaker Endpoint "
    
    print("Get the latest training job name...")
    with open(FILENAME_TRAINING_JOB_NAME) as f:
        training_job_name = f.read()
        
    print("Training job name :", training_job_name)
    
    print("Build the Model...")
    model = TensorFlowModel(
              entry_point=ENTRY_POINT_INFERENCE,
              source_dir=SOURCE_DIR,
              framework_version=FRAMEWORK_VERSION,
              model_data=f"{MODEL_ARTIFACTS_S3_LOCATION}/{training_job_name}/output/model.tar.gz",
              code_location=CUSTOM_CODE_SERVING_UPLOAD_S3_LOCATION,
              name=MODEL_NAME,
              role=ROLE_ARN,
              sagemaker_session=SESS)

    print("Build an endpoint...")
    predictor = model.deploy(endpoint_name=ENDPOINT_NAME, 
                             initial_instance_count=DEPLOY_INSTANCE_COUNT, 
                             instance_type=DEPLOY_INSTANCE_TYPE)
    
    return predictor

    
def main(args):
    # executing function
    if args.mode == "train":
        train()
    elif args.mode == "deploy":
        deploy()
    else:
        raise RuntimeError(f"{args.mode} is not recognized.")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "deploy"]
    )
    args = parser.parse_args()
    main(args)