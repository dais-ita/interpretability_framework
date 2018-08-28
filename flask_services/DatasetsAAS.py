from flask import Flask, Response ,send_file , send_from_directory ,request
import os

import cv2

import base64
from PIL import Image
from StringIO import StringIO

import json

import sys

import random

import numpy as np

import math

import shutil


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


def LoadDatasetTool(dataset_name, model_name = ""):
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
	
	if(model_name == ""):
		training_split_file = dataset_json["default_training_allocation_path"]
		training_split_file_path = os.path.join(datasets_path,"dataset_csvs",training_split_file)
		dataset_tool.ProduceDataFromTrainingSplitFile(training_split_file_path, explicit_path_suffix = dataset_images_dir_path)
	else:
		#TODO allow for passing the model name and load the specific split from the models training(replace the code below with solution)
		training_split_file = dataset_json["default_training_allocation_path"]
		training_split_file_path = os.path.join(datasets_path,"dataset_csvs",training_split_file)
		dataset_tool.ProduceDataFromTrainingSplitFile(training_split_file_path, explicit_path_suffix = dataset_images_dir_path)

	#Code to create new split if required
	#dataset_tool.SplitLiveData(train_ratio=0.8,validation_ratio=0.1,test_ratio=0.1) #splits the live dataset examples in to train, validation and test sets

	return dataset_tool,label_names

def EncodeSpecificImage(dataset_name,image_name):
	dataset_json = GetDatasetJsonFromName(dataset_name)
	
	image_name_split = image_name.split("_")
	image_id = image_name_split[0]
	label = image_name.replace(".jpg","").replace(image_id+"_","")
	
	dataset_path = os.path.join(datasets_path,"dataset_images",dataset_json["folder"])
	label_folder_path = os.path.join(dataset_path,label)
	image_path = os.path.join(label_folder_path,image_name)

	img = cv2.imread(image_path)

	enc_image = encIMG64(img,convert_colour=False)

	return enc_image, label, image_name



def EncodeTestImages(dataset_name,num_images=1, model_name = ""):
	if(not dataset_name in dataset_tools):
		dataset_tool, label_names = LoadDatasetTool(dataset_name,model_name)
		dataset_tools[dataset_name] = {"dataset_tool":dataset_tool,"label_names":label_names}

	label_names = dataset_tools[dataset_name]["label_names"]
	n_classes = len(label_names)
	source = "test"
	print("label_names",label_names)
	x, y, batch = dataset_tools[dataset_name]["dataset_tool"].GetBatch(batch_size = math.ceil(num_images / float(n_classes)) * n_classes,even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = False, batch_source = source, return_batch_data=True)
	print("len(x)",len(x))
	random_i_list = random.sample(list(range(0,len(x))),num_images)

	enc_image_list = []
	y_list = []
	image_name_list = []

	for random_i in random_i_list:

		img_path = list(batch[random_i])[0]
		image_name_list.append( img_path.split("/")[-1] )

		enc_image_list.append( encIMG64(x[random_i]*255,convert_colour=True) )
		y_list.append( y[random_i] )

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


	
	return enc_image_list, y_list, image_name_list



@app.route("/datasets/test_image/<string:dataset_name>", methods=['GET'])
def ServeTestImage(dataset_name):
	enc_x, y, img_name = EncodeTestImages(dataset_name)
	response_dict = {"input":enc_x[0],"ground_truth":y[0], "image_name":img_name[0]}

	return json.dumps(response_dict)


@app.route("/datasets/test_images/<string:dataset_name>", methods=['GET'])
def Serve10TestImages(dataset_name):
	num_images = 10

	enc_x, y, img_name = EncodeTestImages(dataset_name,num_images)
	response_dict = {"input":enc_x,"ground_truth":y, "image_name":img_name}

	return json.dumps(response_dict)


@app.route("/datasets/test_images", methods=['POST'])
def ServeNTestImages():
	raw_json = json.loads(request.data)

	dataset_name = raw_json["dataset_name"]
	num_images = int(raw_json["num_images"])

	enc_x, y, img_name = EncodeTestImages(dataset_name,num_images)
	response_dict = {"input":enc_x,"ground_truth":y, "image_name":img_name}

	return json.dumps(response_dict)



@app.route("/datasets/test_image/specific", methods=['GET'])
def ServeSpecificImages():
	dataset_name = request.args["dataset"]
	image_name = request.args["image_name"]

	enc_x, y, img_name = EncodeSpecificImage(dataset_name,image_name)
	response_dict = {"input":enc_x,"ground_truth":y, "image_name":img_name}

	return json.dumps(response_dict)


@app.route("/datasets/get_available", methods=['GET'])
def GetAvailableDatasetsJson():
	return json.dumps(datasets_json)


def zipdir(path, zip_path):
	shutil.make_archive(zip_path, 'zip', path)
	
@app.route("/datasets/archive", methods=['get'])
def GetModelZip():
	# raw_json = json.loads(request.data)

	dataset_folder_name = request.args.get("dataset_folder_name")
	print("sending:",dataset_folder_name)
	dataset_dir_path = os.path.join(datasets_path,"dataset_images",dataset_folder_name)
	zipped_model_path = os.path.join(datasets_path,"dataset_images","dataset_archives",dataset_folder_name+".zip")
	if(not os.path.exists(zipped_model_path)):
		zipdir(dataset_dir_path, zipped_model_path[:-4])

	return send_file(zipped_model_path, attachment_filename=dataset_folder_name+".zip", as_attachment=True)
	


if __name__ == "__main__":
	print("load dataset jsons")

	#### load dataset json
	data_json_path = os.path.join(datasets_path,"datasets.json")

	datasets_json = None
	with open(data_json_path,"r") as f:
		datasets_json = json.load(f)
	
	print('Starting the API')
	app.run(host='0.0.0.0',port=6001)

