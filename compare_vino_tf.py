"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

import numpy as np

import itertools
from math import exp
import copy

import datetime

import tensorflow as tf


# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-o", "--output", type=str, default=False, required=False,
                        help="Write result to this output-file")
    parser.add_argument("-c", "--cv-output", dest='cv_output', default=False, required=False, action='store_true',
                        help="output in a cv-window")
    parser.add_argument("-fs", "--frame-stats", dest='frame_stats', default=False, required=False, action='store_true',
                        help="Stats are written on the frames")
    parser.add_argument("--no-stream", dest='no_stream', default=False, required=False, action='store_true',
                        help="doesn't write a stream output")
    parser.add_argument("--no-mqtt", dest='no_mqtt', default=False, required=False, action='store_true',
                        help="doesn't publish to mqtt")
    return parser


def connect_mqtt():
    ### Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def draw_masks(result, width, height):
    '''
    Draw semantic mask classes onto the frame.
    '''
    # Create a mask with color by class
    classes = cv2.resize(result[0].transpose((1,2,0)), (width,height), interpolation=cv2.INTER_NEAREST)
    unique_classes = np.unique(classes)
    out_mask = classes * (255/20)
    
    # Stack the mask so FFmpeg understands it
    out_mask = np.dstack((out_mask, out_mask, out_mask))
    # out_mask = np.uint8(out_mask)

    return out_mask, unique_classes

def infer_with_vino(args):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()

    ### Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()

    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    ### Loop until stream is over ###
    while cap.isOpened():

        ### Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        ### Pre-process the image as needed ###
        frame_resized = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        frame_resized = frame_resized.transpose((2,0,1))
        frame_resized = frame_resized.reshape(1, *frame_resized.shape)

        ### Start asynchronous inference for specified request ###
        infer_network.exec_net(frame_resized)

        ### Wait for the result ###
        if infer_network.wait() == 0:
            ### Get the results of the inference request ###
            result = infer_network.get_output()
            
    cap.release()

def read_pb_return_tensors(graph, pb_file, return_elements):

    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)
    return return_elements

def infer_with_tf(args):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()

    ### Load the model through tensorflow ###
    return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
    pb_file         = "./yolov3_coco.pb"
    video_path      = "./docs/images/road.mp4"
    # video_path      = 0
    num_classes     = 80
    input_size      = 416
    graph           = tf.Graph()
    return_tensors  = read_pb_return_tensors(graph, args.pb_file, return_elements)
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()

    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    ### Loop until stream is over ###
    while cap.isOpened():

        ### Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        ### Pre-process the image as needed ###
        frame_resized = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        frame_resized = frame_resized.transpose((2,0,1))
        frame_resized = frame_resized.reshape(1, *frame_resized.shape)

        ### Start asynchronous inference for specified request ###
        infer_network.exec_net(frame_resized)

        ### Wait for the result ###
        if infer_network.wait() == 0:
            ### Get the results of the inference request ###
            result = infer_network.get_output()
            
    cap.release()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Perform inference on the input stream
    infer_on_stream(args)


if __name__ == '__main__':
    main()
