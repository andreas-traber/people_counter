#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from numpy.lib.function_base import append
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### Initialize any class variables desired ###
        pass

    def load_model(self, model, device="CPU", cpu_extension=None):
        ### Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        self.ie = IECore()
        self.net = self.ie.read_network(model=model_xml, weights=model_bin)
        self.exec_network = self.ie.load_network(network=self.net, device_name=device)
        # print(self.ie.query_network(network=self.net, device_name=device))

        # net = IENetwork()
        # print(self.net.layers)

        ### TODO: Check for supported layers ###
        self.output_blobs = next(iter(self.net.outputs))
        # print(self.output_blobs)
        #print(net.outputs)
        ### TODO: Add any necessary extensions ###
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        return

    def get_input_shape(self):
        ### Return the shape of the input layer ###
        return self.net.input_info['inputs'].input_data.shape

    def exec_net(self, image):
        ### Start an asynchronous request ###
        self.exec_network.start_async(request_id=0, inputs={'inputs': image})
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        status = self.exec_network.requests[0].wait(-1)
        return status

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[0].output_blobs

    def get_bbox_size(self, layer_name):
        return 1 + int(self.net.layers[layer_name].params['coords']) + int(self.net.layers[layer_name].params['classes'])

    def get_num_bboxes(self, layer_name):
        return len(self.net.layers[layer_name].params['mask'].split(','))

    def create_anchors(self, layer_name):
        anchors = self.net.layers[layer_name].params['anchors'].split(',')

        self.anchors =  []
        for i in [int(i) for i in self.net.layers[layer_name].params['mask'].split(',')]:
            self.anchors += [ anchors[i * 2], anchors[i * 2 + 1]]
        print(self.anchors)