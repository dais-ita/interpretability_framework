from flask import Flask, Response ,send_file , send_from_directory, request
import os

import cv2

import base64
from PIL import Image
from StringIO import StringIO

import json

import sys

from skimage.segmentation import mark_boundaries

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



#TODO remove dependancy on reloading the model
#add all model folders to sys path to allow for easy import
models_path = os.path.join(base_dir,"models")

model_folders = os.listdir(models_path)

for model_folder in model_folders:
	model_path = os.path.join(models_path,model_folder)
	sys.path.append(model_path)



#add all explanation folders to sys path to allow for easy import
explanations_path = os.path.join(base_dir,"explanations")

explanation_folders = os.listdir(explanations_path)

for explanation_folder in explanation_folders:
	explanation_path = os.path.join(explanations_path,explanation_folder)
	sys.path.append(explanation_path)



app = Flask(__name__)


def readb64(base64_string,convert_colour=True):
    sbuf = StringIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    if(convert_colour):
    	return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
    else:
    	return np.array(pimg) 
def encIMG64(image,convert_colour = False):
    if(convert_colour):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    retval, img_buf = cv2.imencode('.jpg', image)
    
    return base64.b64encode(img_buf)




def DecodeTestImages(images):
	test_images = []

	for string_image in images:
		test_images.append(readb64(string_image))

	return test_images
	


def FilterExplanations(explanations_json,dataset_name,model_name):
	available_explanations = [explanation for explanation in explanations_json["explanations"] if (dataset_name in [dataset["dataset_name"] for dataset in explanation["compatible_datasets"]]) and (model_name in [model["model_name"] for model in explanation["compatible_models"]])]
	print("filtering by :" +dataset_name + " and " + model_name)
	return {"explanations":available_explanations}


@app.route("/explanations/get_available_for_filters/<string:filters>", methods=['GET'])
def GetAvailableExplanationsJSONforFilters(filters):
	print("filters",filters)
	filters_split = filters.split(",")
	dataset_name = filters_split[0]
	model_name = filters_split[1]

	available_explanations = FilterExplanations(explanations_json,dataset_name,model_name)
	
	return json.dumps(available_explanations)


@app.route("/explanations/get_available", methods=['GET'])
def GetAvailableExplanationsJson():
	return json.dumps(explanations_json)


def LoadExplainerFromJson(explanation_json,model_instance):
	ExplanationModule = __import__(explanation_json["script_name"]) 
	ExplanationClass = getattr(ExplanationModule, explanation_json["class_name"])
	
	return ExplanationClass(model_instance)


def LoadModelFromJson(model_json,dataset_json):
	model_name = model_json["model_name"]
	
	ModelModule = __import__(model_json["script_name"]) 
	ModelClass = getattr(ModelModule, model_json["class_name"])
	
	dataset_name = dataset_json["dataset_name"]
	input_image_height = dataset_json["image_y"]
	input_image_width = dataset_json["image_x"]
	input_image_channels = dataset_json["image_channels"]


	n_classes = len(dataset_json["labels"]) 
	
	#TODO need a clever way of handling additonal args
	learning_rate = 0.001
	dropout = 0.25

	additional_args = {"learning_rate":learning_rate,"dropout":dropout}

	trained_on_json = [dataset for dataset in model_json["trained_on"] if dataset["dataset_name"] == dataset_name][0]

	model_path = os.path.join(models_path,model_name,trained_on_json["model_path"])
	model_instance = ModelClass(input_image_height, input_image_width, input_image_channels, n_classes, model_dir=model_path, additional_args=additional_args)
	model_instance.LoadModel(model_path) ## for this model, this call is redundant. For other models this may be necessary. 

	return model_instance



def ImagePreProcess(image):
	#TODO check if this division by 255 is needed
	if(True):
		image = image/255.0
	
	return image.astype(np.float32)


@app.route("/explanations/explain", methods=['POST'])
def Explain():
	raw_json = json.loads(request.data)

	dataset_json = json.loads(raw_json["selected_dataset_json"])
	model_json = json.loads(raw_json["selected_model_json"])
	explanation_json = json.loads(raw_json["selected_explanation_json"])

	input_image = ImagePreProcess(readb64(raw_json["input"],False))

	# dataset_json = json.loads(request.form["selected_dataset_json"])
	# model_json = json.loads(request.form["selected_model_json"])
	# explanation_json = json.loads(request.form["selected_explanation_json"])

	# input_image = ImagePreProcess(readb64(request.form["input"],False))

	
	dataset_name = dataset_json["dataset_name"]
	model_name = model_json["model_name"]
	explanation_name = explanation_json["explanation_name"]

	if(not model_name in loaded_models):
		loaded_models[model_name] = {}

	if(not dataset_name in loaded_models[model_name] ):
		loaded_models[model_name][dataset_name] = LoadModelFromJson(model_json,dataset_json)



	if(not explanation_name in loaded_explanations):
		loaded_explanations[explanation_name] = {}

	if(not model_name in loaded_explanations[explanation_name]):
		loaded_explanations[explanation_name][model_name] = {}

	if(not dataset_name in loaded_explanations[explanation_name][model_name]):
		loaded_explanations[explanation_name][model_name][dataset_name] = LoadExplainerFromJson(explanation_json,loaded_models[model_name][dataset_name])


	explanation_instance = loaded_explanations[explanation_name][model_name][dataset_name]
	
	#TODO allow for better handling of additonal arguments
	additional_args = {"num_samples":100,"num_features":300,"min_weight":0.01}

	explanation_image, explanation_text, prediction, additional_outputs = explanation_instance.Explain(input_image,additional_args=additional_args)
	
	#TODO check if this is needed
	if(True):
		explanation_image_255 = explanation_image*255

	encoded_explanation_image = encIMG64(explanation_image_255,True)

	### test images by displaying pre and post encoding
	display_encoded_image = False

	if(display_encoded_image):
		decoded_image = readb64(encoded_explanation_image)
		cv2_image = decoded_image
		cv2.imshow("image 0",cv2_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	
	labels = [label["label"] for label in dataset_json["labels"]]
	labels.sort()

	print("prediction:"+str(labels[int(prediction)]))#+" - "+labels[prediction)
	# json_data = json.dumps({'prediction': labels[prediction[0]],"explanation_text":explanation_text,"explanation_image":encoded_explanation_image})
	json_data = json.dumps({'prediction': labels[int(prediction)],"explanation_text":explanation_text,"explanation_image":encoded_explanation_image})

	return json_data



if __name__ == "__main__":
	print("load explanations jsons")

	#### load explanations json
	explanations_json_path = os.path.join(explanations_path,"explanations.json")

	explanations_json = None
	with open(explanations_json_path,"r") as f:
		explanations_json = json.load(f)
	

	loaded_models = {}
	loaded_explanations = {}

	print('Starting the API')
	app.run(host='0.0.0.0',port=6201)

