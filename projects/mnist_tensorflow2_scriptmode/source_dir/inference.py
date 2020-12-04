
from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import re
import os
import numpy as np

logging.getLogger().setLevel(logging.INFO)

IMG_SIZE = 28



def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    if context.request_content_type == "text/csv":
        """
            Expect data of the following form : "1.77, 2.0, ..., 1.6\n1.34, 2.9, ..., 4.7"
        """
        tmp = data.read().decode("utf-8").split("\n")
        for i, string in enumerate(tmp):
            array = np.fromstring(string, sep=",")
            array = array.reshape((1, IMG_SIZE, IMG_SIZE, 1))
            if i == 0:
                _input = array
            else:
                _input = np.concatenate([_input, array], axis=0)
        return json.dumps({
            'instances': _input.tolist()
        })
    
    if context.request_content_type == "application/json":
        """
            Expect data of the following form : [[1.33, 4.6, ..., 5.90], [3.5, 4.897, ..., 7.04]]
        """
        tmp = json.loads(data.read().decode("utf-8"))
        for i, e in enumerate(tmp):
            array = np.asarray(e)
            array = array.reshape((1, IMG_SIZE, IMG_SIZE, 1))
            if i == 0:
                _input = array
            else:
                _input = np.concatenate([_input, array], axis=0)
        return json.dumps({
            'instances': _input.tolist()
        })
    
    raise ValueError('{{"Error": "Unsupported content type {}"}}'.format(context.request_content_type or "unknown"))



def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))

    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type
    
    
