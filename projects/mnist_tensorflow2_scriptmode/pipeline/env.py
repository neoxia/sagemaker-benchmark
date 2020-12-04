
# import libraries
import os
import yaml
import datetime
import boto3
import sagemaker


# -------- common variables --------
BOTO3_SESSION = boto3.session.Session(aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                                      aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                                      region_name=os.environ["AWS_DEFAULT_REGION"])
SESS = sagemaker.Session(boto_session=BOTO3_SESSION)
ACCOUNT_ID = SESS.boto_session.client('sts').get_caller_identity().get('Account')
DEFAULT_BUCKET = SESS.default_bucket()

PROJECT_ROOT_PATH = "projects/mnist_tensorflow2_scriptmode"
CUR_TIME = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
SOURCE_DIR = f"{PROJECT_ROOT_PATH}/source_dir"
FILENAME_TRAINING_JOB_NAME="training_job_name.txt"  # used to store the training job and pass it as artifact
ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/service-role/AmazonSageMaker-ExecutionRole-20180124T121654"

with open(f"{PROJECT_ROOT_PATH}/env.yml", "r") as f:
    env = yaml.load(f, Loader=yaml.FullLoader)
PROJECT_ID = env["project_id"]
ENTRY_POINT_TRAIN = env["entry_point_train"]
MODEL_ID = env["model_id"]
PYTHON_VERSION = env["python_version"]
FRAMEWORK_VERSION = env["framework_version"]

# -------- train variables --------
TRAIN_INSTANCE_COUNT = 1
TRAIN_INSTANCE_TYPE = "ml.m4.xlarge"
TRAIN_DATA_S3_LOCATION = f"s3://{DEFAULT_BUCKET}/{PROJECT_ID}/data/train"
TEST_DATA_S3_LOCATION = f"s3://{DEFAULT_BUCKET}/{PROJECT_ID}/data/test"
MODEL_ARTIFACTS_S3_LOCATION = f"s3://{DEFAULT_BUCKET}/{PROJECT_ID}/training"
CUSTOM_CODE_TRAIN_UPLOAD_S3_LOCATION = f"s3://{DEFAULT_BUCKET}/{PROJECT_ID}/training"
BASE_JOB_NAME = MODEL_ID + "-training-job"
HYPERPARAMETERS_FILE_PATH = f"{PROJECT_ROOT_PATH}/pipeline/hyperparameters.json"
DISTRIBUTIONS = {'parameter_server': {'enabled': True}}


# -------- deploy variables --------
ENTRY_POINT_INFERENCE = env["entry_point_inference"]
CUSTOM_CODE_SERVING_UPLOAD_S3_LOCATION = f"s3://{DEFAULT_BUCKET}/{PROJECT_ID}/serving-model"
ENDPOINT_NAME = f"endpoint-{MODEL_ID}-{CUR_TIME}"  
MODEL_NAME = f"model-{MODEL_ID}-{CUR_TIME}"
DEPLOY_INSTANCE_TYPE = "ml.t2.medium"
DEPLOY_INSTANCE_COUNT = 1