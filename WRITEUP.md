# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The process behind converting custom layers involves :
- Generating an the Extension Template Files Using the Model Extension Generator
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
I downloaded this git project https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3.git and used my yolov3 weights and the Pedestrian-Video from the Udacity Project.

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

Each of these use cases would be useful because...

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
