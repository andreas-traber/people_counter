# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The process behind converting custom layers involves :
- Generating an the Extension Template Files Using the Model Extension Generator
- Registering the custom layers as extensions to the Model Optimizer (TensorFlow and Caffe)
- Replacing unsupported subgraph with a different subgraph(TensorFlow only)
- Using Model Optimizer to Generate IR Files Containing the Custom Layer
- Editing the Extractor Extension Template File
- Editing the Operation Extension Template File
- Generating the Model IR Files
- Editing the CPU Extension Template Files
- Compiling the Extension Library
- Adding the CPU-Extension to the Inference Engine


Some of the potential reasons for handling custom layers are:
- When the Model Optimizier doesn't support a layer

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were:
I downloaded this git project https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3.git and used my yolov3 weights and the Pedestrian-Video from the Udacity Project. I added a output for the probability for each frame and the time it took to run. I did the same in my projekt (argument --print-stats) and compared the values in an Excel-Sheet(compare_vino_tf.ods).

The difference between model accuracy pre- and post-conversion was: The model pre-conversion identified 137 more frames with people(overall 1060), where as the converted model found 1 more frame with a person(overall 922). Both using a threshold of 0.5. Changing the threshold to 0.4 and 0.3 for the converted model didn't make much difference. Therefore the accuracy of the converted model is not sufficent.

The size of the model pre- and post-conversion was: The original yolov3.weights has 248 MB, the coverted bin+xml-files have nearly the same size.

The inference time of the model pre- and post-conversion was: The interference Engine took around 335 seconds, for the whole video. The downloaded Tensorflow-Project took around 436 seconds.

When comparing the FPS(tab FPS in compare_vino_tf.ods) the original model has average FPS of ~ 3.20 and the converted model has average FPS of ~ 4.55.

As suggested by my Reviewer, I tried person-detection-retail-0013(FPS32) from the Pre-Trained Models. Detection Rate was about the same as the original Yolov3-Model. Size of xml+bin-Files is only 3.2 MB. It took about 22 seconds for the whole video, with average FPS of about 64. 



## Assess Model Use Cases

Some of the potential use cases of the people counter app are:
-You could use this counter for events, where there is a limit on how many people are allowed in a room/builidng and count the people, who went in, and substract the people who left.
-For polling stations, you could count, how many people went voting and compare it to the number of casted votes. 

Each of these use cases would be useful because:
they outsource boring repetitive work to a machine.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...
Lighning:
    e.g. in dark rooms, people with dark clothes are harder to detect, since there are no clear edges between background and person
Model accurary:
    means, how well the model recognizes the trained classes
Camera focal length/image size:
    A camera with a high focal length is better in recognizing specific objects, whereas a camera with a low focal length can cover bigger spaces.

## Model Research

### Yolov3

#### Convert to Intermediate Representation
Download the Class Names 
`wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names`

Download the model
`wget https://pjreddie.com/media/files/yolov3.weights`

Use this project to convert to PB(Tensoerflow)-format
`git clone https://github.com/mystic123/tensorflow-yolo-v3.git`

`python3 tensorflow-yolo-v3/convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3.weights`

Use OpenVino Model Optimzier to convert to Intermediate Representation
`python3 ../../openvino/model-optimizer/mo.py --input_model frozen_darknet_yolov3_model.pb --tensorflow_use_custom_operations_config ../../openvino/model-optimizer/extensions/front/tf/yolo_v3.json --batch 1`

#### Performance
Even though the original Yolov3-model seems to be sufficent for this application, the converterted model missed a lot frames(see Comparing Model Performance) and therefor didn't meet the requirements for this project.

### person-detection-retail-0013

#### Download
Just use the downloader from the model-zoo
`python3 downloader.py --name person-detection-retail-0013 --precision FP32 -o ~/python_projects/udacity/people_counter/model/`

#### Performance
This model met the requirements of the Project. Even thought, the original Yolov3-Model had about the same accuracy, this model from the model zoo is much smaller and much faster.


