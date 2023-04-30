# DATASCI 3ML3 - Machine Learning Project

## Introduction
As a part of the DATASCI 3ML3 Course offered at McMaster University, I was tasked with 
defining a machine learning related project and implementing it. I chose to work on an image
segmentation network that would be able to semanticallly segment scenes to provide context
for autonomous vehicles. The dataset I used was the KITTI dataset, which is an open-source dataset
available at https://www.cvlibs.net/datasets/kitti-360/. 

## How to Use
The dataset is not included in this repository, but can be downloaded from the link above.
The dataset is quite large, so I would recommend using a subset of the data to train the model.
The main entry point for the project is the `main.py` file. The file contains a number of functions
that can be used to train and test the model. The `main` function can be used to train the model and 
the 'test_checkpoint' function can be used to test the model.
The 'params.py' file contains a number of parameters that can be used to configure the model training pipeline and inferencing.


## Sample output from trained model
A sample output from the trained model is shown below. The model was trained on a subset of the KITTI dataset
![Sample Output](/sample_image.png)

## Pre-trained Model
A pre-trained model is available in the `train_output` folder along with a video showcasing model outputs. 

### Final Report
Check out the final report for this project [here](Final%20Report%20-%20alamo2.pdf).

