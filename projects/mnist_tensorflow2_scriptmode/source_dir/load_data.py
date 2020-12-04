
from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import re
import os
import pandas as pd
import struct
from array import array
import numpy as np

logging.getLogger().setLevel(logging.INFO)

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)
    
    

def get_datasets(train_data_path, test_data_path):
    """
        Function used to get your data
        
        Arguments:
            train_data_path (str): path string to a directory where lies your train data files
            test_data_path (str): path string to a directory where lies your test data files
            
        Returns:
            tuple of four numpy arrays in this order : x_train, y_train, x_test, y_test.
            x_train has a shape of (NUM_ELEMENT_TRAIN, IMG_SIZE, IMG_SIZE, 1). The one at the end is for the number of channel.
            In this case the number of channel is one since the images are in grayscale.
    """
    # get the data
    try:
        # decompress the .gz files
        os.system("gunzip " + train_data_path + "/*")
        os.system("gunzip " + test_data_path + "/*")
        
        # load the images
        mdl = MnistDataloader(training_images_filepath=os.path.join(train_data_path, "train-images-idx3-ubyte"),
                              training_labels_filepath=os.path.join(train_data_path, "train-labels-idx1-ubyte"),
                              test_images_filepath=os.path.join(test_data_path, "t10k-images-idx3-ubyte"),
                              test_labels_filepath=os.path.join(test_data_path, "t10k-labels-idx1-ubyte"))
        (x_train, y_train), (x_test, y_test) = mdl.load_data()
    except:
        raise RuntimeError("Error when trying to load the data.")
    
    return np.expand_dims(np.array(x_train), -1), np.array(y_train), np.expand_dims(np.array(x_test), -1), np.array(y_test)
    