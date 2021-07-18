# HUMAN SEMANTIC SEGMENTATION


## Dataset

The dataset was acquired from https://github.com/VikramShenoy97/Human-Segmentation-Dataset.

It is a small dataset comprising 290 images and the corresponding masks for semantic segmentation projects.

In each mask humans were labeled depending on their positions on the corresponding images.
Example:  
  
![training image](./Human-Segmentation-Dataset/Training_Images/21.jpg) ![mask image](./Human-Segmentation-Dataset/Ground_Truth/21.png)
<br/><br/>
## Goal

The goal of this small project was to showcase a complete end-to-end solution:
- Gathering data
- Preprocessing the data
- Setting a Deep Learning architecture for a semantic segmentation model to be trained
- Train the model
- Assess trained model main metrics
- Create inferences on unseen data using the trained model
- Saving the model for future usage


## Requirements

To install all libraries required:

pip3 install -r requirements.txt

This project's jupyter notebook was run on a GPU GeForce GTX 1060 6GB in a Ubuntu 20.04 OS using Python 3.8.5.