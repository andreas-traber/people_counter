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
                        help="output-file")
    parser.add_argument("-c", "--cv-output", type=bool, default=False, required=False,
                        help="output-file")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
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
    image_flag = False

    # Check if the input is a webcam
    if args.input == 'CAM':
        args.input = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') or args.input.endswith('.png'):
        image_flag = True

    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    width = int(cap.get(3))
    height = int(cap.get(4))
    #out = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(cv2.CAP_PROP_FPS), (width,height))
    frame_duration =1/cap.get(cv2.CAP_PROP_FPS)
    people_count = 0
    prev_person_box_final = None
    last_frame = datetime.datetime.now()
    stats={'person': {'count': 0 , 'total': 0, 'duration': 0.0}}
    skip_frames = 100
    frame_cnt = 0
    ### TODO: Loop until stream is over ###
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        frame_cnt += 1
        if frame_cnt<skip_frames:
            continue
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
            ### TODO: Extract any desired stats from the results ###
            person_box = []
            person_confidence = []
            stop_frame = False
            for layer_name, blob in result.items():
                bbox_size = infer_network.get_bbox_size(layer_name)
                res=blob.buffer[0]
                for row, col, n in  itertools.product(range(res.shape[1]), range(res.shape[2]), range(infer_network.get_num_bboxes(layer_name))):
                    bbox = res[n*bbox_size:(n+1)*bbox_size, row, col]
                    # only need person class probability
                    bbox = bbox[:6]
                    if bbox[4]>args.prob_threshold and bbox[5]>args.prob_threshold:
                        x = (col + bbox[0]) / res.shape[1]
                        y = (row + bbox[1]) / res.shape[2]
                        width_ = exp(bbox[2])
                        height_ = exp(bbox[3])
                        infer_network.create_anchors(layer_name)
                        width_ = width_ * float(infer_network.anchors[2 * n]) / float(net_input_shape[2])
                        height_ = height_ * float(infer_network.anchors[2 * n + 1]) / float(net_input_shape[3])
                        xmin = int((x - width_ / 2) * width)
                        ymin = int((y - height_ / 2) * height)
                        xmax = int(xmin + width_ * width)
                        ymax = int(ymin + height_ * height)
                        person_box.append([xmin, ymin, xmax, ymax])
                        person_confidence.append(float(bbox[5]))
                
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            if len(person_box)>0:
                nms_indexes = cv2.dnn.NMSBoxes(person_box, person_confidence, args.prob_threshold, args.prob_threshold)
                person_box_final = [person_box[i] for i in nms_indexes.flatten()]
                stats['person']['count'] = len(person_box_final)
                if prev_person_box_final:
                    person_box_combined = prev_person_box_final+person_box_final
                    nms_indexes_comb = cv2.dnn.NMSBoxes(person_box_combined, [1 for _ in range(len(person_box_combined))], 0.9, 0.1)
                    stats['person']['total'] += len(nms_indexes_comb)-len(person_box_final)
                    # same person as last seen one ?
                    if len(nms_indexes_comb)>len(person_box_final):
                        #stop_frame = True
                        person_box_comb= [person_box_combined[i] for i in nms_indexes_comb.flatten()]
                        stats['person']['duration'] = 0.000
                    else:
                        stats['person']['duration'] = round(stats['person']['duration']  + frame_duration, 3)
                else:
                    stats['person']['total'] += len(person_box_final)
                prev_person_box_final = copy.copy(person_box_final)
                for rect in person_box_final:
                    cv2.rectangle(frame,(rect[0],rect[1]),(rect[2],rect[3]),[0,0,255],6)
                if stop_frame:
                    for rect in person_box_comb:
                        cv2.rectangle(frame,(rect[0],rect[1]),(rect[2],rect[3]),[0,255,0],6)
            else:
                stats['person']['count'] = 0

            FPS = round(1000000 / (datetime.datetime.now() - last_frame).microseconds, 2)
            last_frame = datetime.datetime.now()
            stats_string = ['Frame: %s' % frame_cnt,
                            'FPS: %s' % FPS, 
                            'Count: %s' % stats['person']['count'],
                            'Total: %s' % stats['person']['total'],
                            'Duration(s): %s' % stats['person']['duration']]
            for i in range(len(stats_string)):
                cv2.putText(frame, stats_string[i], (15, 20+i*20), cv2.FONT_HERSHEY_COMPLEX, 0.75, (200, 10, 10), 2)
            #out.write(frame)
            #cv2.imshow('test', frame)
            """if stop_frame:
                cv2.waitKey()"""
            key = cv2.waitKey(1)

            if key in {ord("q"), ord("Q"), 27}: # ESC key
                break
            client.publish('person', json.dumps(stats['person']))
            client.publish('person/duration', json.dumps(stats['person']['duration']))
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        if image_flag:
            cv2.imwrite('output_image.jpg', frame)
    cap.release()
    
    out.release()    
    client.disconnect()

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
