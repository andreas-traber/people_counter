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
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = None

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

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model)
    net_input_shape = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    width = int(cap.get(3))
    height = int(cap.get(4))
    i=0
    ### TODO: Loop until stream is over ###
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break

        ### TODO: Pre-process the image as needed ###
        frame_resized = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        frame_resized = frame_resized.transpose((2,0,1))
        frame_resized = frame_resized.reshape(1, *frame_resized.shape)

        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(frame_resized)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()
            """
            rectancles = [[int(x[3]*width),
                           int(x[4]*height),
                           int(x[5]*width),
                           int(x[6]*height)]
                          for x in result[0][0] if x[2]>prob_threshold]
            for rect in rectancles:
                cv2.rectangle(frame,(rect[0],rect[1]),(rect[2],rect[3]),[255,0,0],6)
            cv2.imshow('test', frame)
            cv2.waitKey()
            #"""
            ### TODO: Extract any desired stats from the results ###
            person_list = []
            for layer_name, blob in result.items():
                bbox_size = infer_network.get_bbox_size(layer_name)
                res=blob.buffer[0]
                for row, col, n in  itertools.product(range(res.shape[1]), range(res.shape[2]), range(infer_network.get_num_bboxes(layer_name))):
                    bbox = res[n*bbox_size:(n+1)*bbox_size, row, col]
                    # only need person class
                    bbox = bbox[:6]
                    if bbox[4]>args.prob_threshold and bbox[5]>args.prob_threshold:
                        person_list.append(bbox)
            print(person_list)
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

        ### TODO: Send the frame to the FFMPEG server ###

        ### TODO: Write an output image if `single_image_mode` ###
        if i<0:
            i+=1
        else:
            break

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
