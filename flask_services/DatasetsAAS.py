from flask import Flask, Response ,send_file , send_from_directory
import os

import cv2

import base64
from PIL import Image
from StringIO import StringIO

import json

import sys

import random

import numpy as np



### Setup Sys path for easy imports
# base_dir = "/media/harborned/ShutUpN/repos/dais/p5_afm_2018_demo"
# base_dir = "/media/upsi/fs1/harborned/repos/p5_afm_2018_demo"

def GetProjectExplicitBase(base_dir_name="p5_afm_2018_demo"):
	cwd = os.getcwd()
	split_cwd = cwd.split("/")

	base_path_list = []
	for i in range(1, len(split_cwd)):
		if(split_cwd[-i] == base_dir_name):
			base_path_list = split_cwd[:-i+1]

	if(base_path_list == []):
		raise IOError('base project path could not be constructed. Are you running within: '+base_dir_name)

	base_dir_path = "/".join(base_path_list)

	return base_dir_path

base_dir = GetProjectExplicitBase(base_dir_name="p5_afm_2018_demo")

#add dataset folder to sys path to allow for easy import
datasets_path = os.path.join(base_dir,"datasets")
sys.path.append(datasets_path)
###

dataset_tools = {}


#import dataset tool
from DatasetClass import DataSet




app = Flask(__name__)


def readb64(base64_string):
    sbuf = StringIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

def encIMG64(image,convert_colour = False):
    if(convert_colour):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    retval, img_buf = cv2.imencode('.jpg', image)
    
    return base64.b64encode(img_buf)


def GetDatasetJsonFromName(dataset_name):
	return [dataset for dataset in datasets_json["datasets"] if dataset["dataset_name"] == dataset_name][0]


def LoadDatasetTool(dataset_name):
	dataset_json = GetDatasetJsonFromName(dataset_name)
	### gather required information about the dataset
	file_path = dataset_json["ground_truth_csv_path"]
	image_url_column = "image_path"
	ground_truth_column = "label"
	label_names = [label["label"] for label in dataset_json["labels"]] # gets all labels in dataset. To use a subset of labels, build a list manually

	### instantiate dataset tool
	csv_path = os.path.join(datasets_path,"dataset_csvs",file_path)
	dataset_images_dir_path =  os.path.join(datasets_path,"dataset_images")
	dataset_tool = DataSet(csv_path,image_url_column,ground_truth_column,explicit_path_suffix =dataset_images_dir_path) #instantiates a dataset tool
	dataset_tool.CreateLiveDataSet(dataset_max_size = -1, even_examples=True, y_labels_to_use=label_names) #creates an organised list of dataset observations, evenly split between labels
	
	#TODO We should save the split during training/a global set to be used in all training and we should load the same split when reusing the dataset
	dataset_tool.SplitLiveData(train_ratio=0.8,validation_ratio=0.1,test_ratio=0.1) #splits the live dataset examples in to train, validation and test sets

	return dataset_tool,label_names


def EncodeTestImages(dataset_name,num_images=1):
	if(not dataset_name in dataset_tools):
		dataset_tool, label_names = LoadDatasetTool(dataset_name)
		dataset_tools[dataset_name] = {"dataset_tool":dataset_tool,"label_names":label_names}

	label_names = dataset_tools[dataset_name]["label_names"]
	source = "test"

	x, y = dataset_tools[dataset_name]["dataset_tool"].GetBatch(batch_size = len(label_names),even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = False, batch_source = source)

	random_i = random.randint(0,len(x)-1)

	enc_image = encIMG64(x[random_i]*255,convert_colour=True)


	### test images by displaying pre and post encoding
	display_example_image = False

	if(display_example_image):
		cv2_image = cv2.cvtColor(x[random_i], cv2.COLOR_RGB2BGR)
		cv2.imshow("image 0",cv2_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	display_encoded_image = False

	if(display_encoded_image):
		decoded_image = readb64(enc_image)
		cv2_image = decoded_image
		cv2.imshow("image 0",cv2_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


	
	return enc_image, y[random_i]



@app.route("/datasets/test_image/<string:dataset_name>", methods=['GET'])
def ServeTestImages(dataset_name):
	enc_x, y = EncodeTestImages(dataset_name)
	response_dict = {"input":enc_x,"ground_truth":y}

	return json.dumps(response_dict)


@app.route("/datasets/get_available", methods=['GET'])
def GetAvailableDatasetsJson():
	return json.dumps(datasets_json)



if __name__ == "__main__":
	print("load dataset jsons")

	#### load dataset json
	data_json_path = os.path.join(datasets_path,"datasets.json")

	datasets_json = None
	with open(data_json_path,"r") as f:
		datasets_json = json.load(f)
	
	print('Starting the API')
	app.run(host='0.0.0.0',port=6001)

