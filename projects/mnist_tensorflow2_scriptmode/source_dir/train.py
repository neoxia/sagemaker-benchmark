
from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import re
import os
import shutil

# --------- INSTALL REQUIREMENTS ----------
print("------ INSTALL REQUIREMENTS ------")
os.system("pip install --upgrade pip")
os.system("pip install -r requirements.txt")

from model import keras_model_fn
from load_data import get_datasets

logging.getLogger().setLevel(logging.INFO)


def main(args):

    logging.info("------ Getting data ------")
    x_train, y_train, x_test, y_test = get_datasets(train_data_path=args.train, test_data_path=args.test)

    logging.info("------ Configuring model ------")
    model = keras_model_fn(vars(args))

    logging.info("------ Executing training ------")
    model.fit(x=x_train,
              y=y_train,
              epochs=int(args.epochs),
              batch_size=int(args.batch_size),
              verbose=0)

    score = model.evaluate(x_test,
                           y_test,
                           verbose=0)

    logging.info("-------- Scores --------")
    logging.info('Test loss:{}'.format(score[0]))
    logging.info('Test accuracy:{}'.format(score[1]))
    
    logging.info("-------- Storing models & information --------")
    # save the model
    model.save(os.path.join(args.model_output_dir, '000000001'), 'model.h5')
    # save the score
    with open(os.path.join(args.model_output_dir, "score.txt"), "w+") as f:
        f.write("Loss Test = " + str(score[0]))
        f.write("Accuracy Test = " + str(score[1]))
    # save the hyperparameters
    json.dump(vars(args), open(os.path.join(args.model_output_dir, "parameters_training.json"), "w"), indent=0)

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # ----------- SAGEMAKER CONTAINER PARAMETERS -----------
    parser.add_argument(
        '--train',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_TRAIN'),
        help='The directory where the input train data is stored.')
    parser.add_argument(
        '--validation',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_VALIDATION'),
        help='The directory where the input test data is stored.')
    parser.add_argument(
        '--test',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_TEST'),
        help='The directory where the input eval data is stored.')
    parser.add_argument(
        '--model_dir',
        type=str,
        help="The directory of the model in S3"
    )
    parser.add_argument(
        '--model_output_dir',
        type=str,
        default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument(
        '--output-dir',
        type=str,
        default=os.environ.get('SM_OUTPUT_DIR'))
    parser.add_argument(
        '--data-config',
        type=json.loads,
        default=os.environ.get('SM_INPUT_DATA_CONFIG')
    )
    parser.add_argument(
        '--fw-params',
        type=json.loads,
        default=os.environ.get('SM_FRAMEWORK_PARAMS')
    )
    
    # ----------- HYPERPARAMETERS -----------
    parser.add_argument('--input-shape')
    parser.add_argument('--num-classes')
    parser.add_argument('--epochs')
    parser.add_argument('--batch-size')
    parser.add_argument('--learning-rate')
    parser.add_argument('--optimizer')
    parser.add_argument('--momentum')
    parser.add_argument('--weight-decay')
    parser.add_argument('--l2-regul')
    parser.add_argument('--dropout')
    parser.add_argument('--cnn')
    parser.add_argument('--dense')
    
    args = parser.parse_args()
    main(args)
