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



#add dataset folder to sys path to allow for easy import
datasets_path = os.path.join(base_dir,"datasets")
sys.path.append(datasets_path)

#import dataset tool
from DatasetClass import DataSet


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

def LoadDatasetFromJson(dataset_json,model_name = ""):
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


def CreateAttributionMap(attribution_slice,slice_weights):
	output_map = np.array(attribution_slice).astype(np.float32)

	for region_weight in slice_weights:
		# print(region_weight[0],region_weight[1])
		output_map[output_map == region_weight[0]] = region_weight[1]

	return output_map


@app.route("/explanations/attribution_map", methods=['POST'])
def GetAttributionMap():
	raw_json = json.loads(request.data)

	if(not "attribution_slices" in raw_json):
		json_data = json.dumps({})

		return json_data

	attribution_slices = json.loads(raw_json["attribution_slices"])
	attribution_slice_weights = json.loads(raw_json["attribution_slice_weights"])

	attribution_map = CreateAttributionMap(attribution_slices,attribution_slice_weights)

	json_data = json.dumps({'attribution_map': attribution_map.tolist()})

	return json_data

	

@app.route("/explanations/explain", methods=['POST'])
def Explain():
	raw_json = json.loads(request.data)

	print(raw_json.keys())

	print(type(raw_json['selected_dataset_json']))
	if(isinstance(raw_json['selected_dataset_json'],str) or isinstance(raw_json['selected_dataset_json'],unicode) ):
		dataset_json = json.loads(raw_json['selected_dataset_json'])
	else:
		dataset_json = raw_json['selected_dataset_json']

	if(isinstance(raw_json['selected_model_json'],str) or isinstance(raw_json['selected_model_json'],unicode)):
		model_json = json.loads(raw_json['selected_model_json'])
	else:
		model_json = raw_json['selected_model_json']

	if(isinstance(raw_json['selected_explanation_json'],str) or isinstance(raw_json['selected_explanation_json'],unicode)):
		explanation_json = json.loads(raw_json['selected_explanation_json'])
	else:
		explanation_json = raw_json['selected_explanation_json']

	input_image = ImagePreProcess(readb64(raw_json["input"],convert_colour=False))

	dataset_name = dataset_json["dataset_name"]
	model_name = model_json["model_name"]
	explanation_name = explanation_json["explanation_name"]


	if(dataset_name not in loaded_dataset_tools):
		loaded_dataset_tools[dataset_name]={}
		loaded_dataset_tools[dataset_name]["dataset_tool"],loaded_dataset_tools[dataset_name]["label_names"] = LoadDatasetFromJson(dataset_json)


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


	load_dataset_images = False #TODO turn this on and off automatically (for actual use, set this to True unless you don't plan to use influence functions)

	if(load_dataset_images):
		# load training images for influence functions
		if(dataset_name not in loaded_training_images):
			loaded_training_images[dataset_name] = {}
			source = "train"
			label_names = loaded_dataset_tools[dataset_name]["label_names"]
			loaded_training_images[dataset_name]["train_x"], loaded_training_images[dataset_name]["train_y"], loaded_training_images[dataset_name]["batch"] = loaded_dataset_tools[dataset_name]["dataset_tool"].GetBatch(batch_size = -1,even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source, return_batch_data=True)

			print("loaded_training_images[dataset_name]['train_x'].shape",loaded_training_images[dataset_name]["train_x"].shape)
		
		#TODO allow for better handling of additonal arguments, currently additional arguments for ALL explanations must be placed here
		additional_args = {
		"num_samples":100,
		"num_features":300,
		"min_weight":0.0001, 
		"model_name":model_name, 
		"dataset_name":dataset_name, 
		"num_background_samples":50,
		"train_x":loaded_training_images[dataset_name]["train_x"],
		"train_y":loaded_training_images[dataset_name]["train_y"],
		"max_n_influence_images":9
		}
	else:
		additional_args = {
		"num_samples":100,
		"num_features":300,
		"min_weight":0.0001, 
		"model_name":model_name, 
		"dataset_name":dataset_name, 
		"num_background_samples":50,
		"max_n_influence_images":9
		}
	print(np.amax(input_image))
	display_explanation_input = False
	if(display_explanation_input):
		cv2_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
		cv2.imshow("image: input_image",cv2_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	explanation_image, explanation_text, prediction, additional_outputs = explanation_instance.Explain(input_image,additional_args=additional_args)
	

	##### testing attribution maps
	# if("attribution_slices" in additional_outputs.keys() and "attribution_slice_weights" in additional_outputs.keys() ):
	# 	attribution_map = CreateAttributionMap(additional_outputs["attribution_slices"],additional_outputs["attribution_slice_weights"])
	# # print("")
	# print(attribution_map)
	# print("")

	#TODO check if this is needed
	if(explanation_image.max() <=1):
		explanation_image_255 = explanation_image*255
	else:
		explanation_image_255 = explanation_image

	encoded_explanation_image = encIMG64(explanation_image_255,False)

	### test images by displaying pre and post encoding
	display_encoded_image = False

	if(display_encoded_image):
		decoded_image = readb64(encoded_explanation_image,convert_colour=True)
		cv2_image = decoded_image
		cv2.imshow("image 0",cv2_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	
	labels = [label["label"] for label in dataset_json["labels"]]
	labels.sort()

	print("prediction:"+str(labels[int(prediction)]))#+" - "+labels[prediction)
	# json_data = json.dumps({'prediction': labels[prediction[0]],"explanation_text":explanation_text,"explanation_image":encoded_explanation_image})
	json_data = json.dumps({'prediction': labels[int(prediction)],"explanation_text":explanation_text,"explanation_image":encoded_explanation_image, "additional_outputs":additional_outputs})

	return json_data



if __name__ == "__main__":
	print("load explanations jsons")

	#### load explanations json
	explanations_json_path = os.path.join(explanations_path,"explanations.json")

	explanations_json = None
	with open(explanations_json_path,"r") as f:
		explanations_json = json.load(f)
	
	loaded_dataset_tools = {}
	loaded_training_images = {}
	loaded_models = {}
	loaded_explanations = {}

	print('Starting the API')
	app.run(host='0.0.0.0',port=6201)

