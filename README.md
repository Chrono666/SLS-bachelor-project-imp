# Frontend for Trading Service - Distributed System School Project

## Table of contents

* [General information](#general-information)
* [Project information](#project-information)
* [Setup](#setup)
* [Notebook description](#notebook-description)

## General information

This project is the technical implementation of the bachelor thesis "In-Situ detection of printing defects during a SLS manufacturing process using deep learning".
A disadvantage of the layer wise manufacturing of SLS is that the component quality can not be assured without destroying 
or scanning the component. For this purpose this thesis uses CNNs on images extracted from the printing processes to detect
curling of the component during the printing process. Curling happens if the component is subjected to hight temperature 
differences during the manufacturing which create residual stress inside it and lead to a curling of the component.
This curling results in the component to stick out of the powder bed and can lead, in the worst case, to the 
recoater to rip the component out.

## Project information

Three different CNN architectures (VGG-16, Xception, ResNet50) are used to train models and are evaluated on a test set, which was not part of the
training. These architectures were pretrained on the imagenet data-set and used for transfer learning by defining a custom
output layer which uses a sigmoid activation function to output a binary classification of the input (1 for OK, 0 for DEFECT).
Afterwards hyperparameter tuning was done to try and increase the performance of the best performing CNN architecture. 
For this purpose and to keep the area of concerns separate multiple jupyter-notebooks were created. 


## Setup

This project was created with python and jupyter notebook. For this purpose a virtual environment was created with anaconda 
and most of the packages were installed through conda, for some packages pip was used. The environment can be found in the env
folder of the project (currently only linux supported).  

In the comand line move to the env folder of the project and use the following commands:
```
$ conda env create -f sls-linux.yml
$ conda activate sls-linux
```  
or use the graphical interface (anaconda-navigator) to import it.  

Use the `jupyter-notebook` for single notebook or `jupyter-lab` to open the whole project in the jupyter-lab environment.

#### NOTE: 
This project also uses the GPU to increase computational speed and performance. What kind of device your operating system provides
is displayed by `tf.config.get_visible_devices()` and is CPU by default. To enable GPU support please follow the instructions
on https://www.tensorflow.org/install/gpu

## Notebook description  

Steps and functionality of the code implemented in the single notebooks is explained in them. And they are seperated into 
different areas of concerns.  

### model_trainer

- loads data set from specified path
- Preprocess images through the ImageDataGenerator
- builds the base_model using the defined CNN architecture (VGG-16, Xception, ResNet50)
- builds the complete model by implementing a custom output layer onto the base_model
- compiles the model and trains it with the base_model layers frozen (step 1 of transfer learning)
- compiles the model again and trains it with the base_model layers unfrozen (step 2 of transfer learning)
- evaluates model on test set which was not used in training and prints the defined metrics
- saves model as json and its weights as h5 to load it for predictions  

### model_predicter

- loads the defined model from the path
- uses loaded model to classify single defined image
- uses loaded model to classify multiple defined images 

### model_grad_cam

- implements grad cam class
  - used to take the activations of the las convolutional layer of the defined model
  - and creates a heatmap of the parts of the image which lead to the activations
- loads defined model from path for grad cam use
- uses loaded model and grad cam class to create heatmap of single image
  - output three images: original, heatmap, heatmap overlay
- uses loaded model and grad cam on multiple images defined by the specified path
- and saves them in the project root folder
  - output original, heatmap overlay

### model_tuner

- builds model for hyperparameter tuning
- uses keras tuner and hyperband search to train model with different parameters
- saved as trials and outputs the parameters with the best results

### model_trainer_multiple

- same as model_trainer notebook
- used to train multiple models after another during nighttime