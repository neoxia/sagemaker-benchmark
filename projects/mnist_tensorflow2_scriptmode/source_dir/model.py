
from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import re
import os

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, Conv2D, MaxPool2D, BatchNormalization, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2

logging.getLogger().setLevel(logging.INFO)

def keras_model_fn(hyperparameters):
    """keras_model_fn receives hyperparameters from the training job and returns a compiled keras model.
    The model is transformed into a TensorFlow Estimator before training and saved in a
    TensorFlow Serving SavedModel at the end of training.
    """
    print("hyperparameters", hyperparameters)
    _input = Input(shape=eval(hyperparameters["input_shape"]))
    x = _input
    # cnn step
    for _cnn in json.loads(hyperparameters["cnn"].replace("\'", "\"")):
        x = Conv2D(_cnn["filters"], eval(_cnn["kernel"]), activation=_cnn["activation"], padding=_cnn["padding"],
                   kernel_regularizer=l2(hyperparameters["l2_regul"]), bias_regularizer=l2(hyperparameters["l2_regul"]))(x)
        x = BatchNormalization()(x)
        x = MaxPool2D((2, 2))(x)
    # dense step
    x = Flatten()(x)
    for _dense in json.loads(hyperparameters["dense"].replace("\'", "\"")):
        x = Dense(_dense["units"], activation=_dense["activation"],
                 kernel_regularizer=l2(hyperparameters["l2_regul"]), bias_regularizer=l2(hyperparameters["l2_regul"]))(x)
        x = Dropout(float(hyperparameters["dropout"]))(x)
    _output = Dense(int(hyperparameters["num_classes"]), activation="softmax")(x)
    
    # generate the model
    model = Model(inputs=_input, outputs=_output)
    
    print(model.summary())

    # optimizer choice
    if hyperparameters["optimizer"].lower() == "sgd":
        opt = SGD(lr=float(hyperparameters["learning_rate"]), decay=float(hyperparameters["weight_decay"]), momentum=float(hyperparameters["momentum"]))
    elif hyperparameters["optimizer"].lower() == "rmsprop":
        opt = RMSprop(lr=float(hyperparameters["learning_rate"]), decay=float(hyperparameters["weight_decay"]))
    else:
        opt = Adam(lr=float(hyperparameters["learning_rate"]), decay=float(hyperparameters["weight_decay"]))
            
    # compile the model
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=opt,
                  metrics=["accuracy"])
    return model