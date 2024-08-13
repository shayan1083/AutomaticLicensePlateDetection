Project to detect and read a license plate on a car.

I did not upload the augmented images or the model that reads the license plate because they were too large. 


create venv:
python3 -m venv venv

activate venv:
source venv/bin/activate

in jupyter notebook, select kernel of venv

license plate detection based on this: https://www.kaggle.com/code/aslanahmedov/automatic-number-plate-recognition

detect_read.ipynb: detect license plate and test neural networks to read it

detect_read.py: detect license plate and read it with final neural net. Working on converting this to a Flask app 

detect.ipynb: train and test yolo model

generate_data.ipynb: generate augmentations to create dataset

get_data_from_img.ipynb: extract each character from an image containing every character for a specific font

data.yaml: Configuration file for YOLO model

datasets.txt: Listing the transformations that were done to obtain each dataset of augmented characters
