from flask import Flask, Response ,send_file , send_from_directory,request
import os

import cv2

import base64
from PIL import Image
from StringIO import StringIO

import json

import sys

import numpy as np

import zipfile


def zipdir(path, zip_path):
    ziph = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

    ziph.close()

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


#add all model folders to sys path to allow for easy import
models_path = os.path.join(base_dir,"models")

model_folders = os.listdir(models_path)

for model_folder in model_folders:
	model_path = os.path.join(models_path,model_folder)
	sys.path.append(model_path)




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


def DecodeTestImages(images):
	test_images = []

	for string_image in images:
		test_images.append(readb64(string_image))

	return test_images
	



def GetModelJsonFromName(model_name):
	return [model for model in models_json["models"] if model["model_name"] == model_name][0]


def LoadModelFromName(model_name,dataset_json):
	model_json = GetModelJsonFromName(model_name)
	
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

def FilterModelByDataset(models_json,dataset_name):
	available_models = [model for model in models_json["models"] if dataset_name in [dataset["dataset_name"] for dataset in model["trained_on"]] ]
	print("filtering by :" +dataset_name)
	return {"models":available_models}


@app.route("/models/get_available_for_dataset/<string:dataset_name>", methods=['GET'])
def GetAvailableModelsJSONforDataset(dataset_name):
	available_models = FilterModelByDataset(models_json,dataset_name)
	
	return json.dumps(available_models)


@app.route("/models/get_available", methods=['GET'])
def GetAvailableModelsJson():
	return json.dumps(models_json)


def PredictImagePreProcess(image):
	if(True): #TODO 'cleverly' decide whether this should happen
		image = image/255.0
	
	# Add any 'always happen' preprocessing steps e.g. check size and resize if needed etc

	return image.astype(np.float32)

@app.route("/models/archive", methods=['get'])
def GetModelZip():
	# raw_json = json.loads(request.data)

	dataset_name = request.args.get("dataset_name")
	model_name = request.args.get("model_name")

	# dataset_name = json.loads(raw_json['selected_dataset_name'])
	# model_name = json.loads(raw_json['selected_model_name'])

	trained_model_file_name = dataset_name.replace(" ","_")+".zip"

	trained_model_dir_path = os.path.join(models_path,model_name,"saved_models",dataset_name.replace(" ","_"))
	zipped_model_path = os.path.join(models_path,model_name,"saved_models","zipped_models",trained_model_file_name)
	if(not os.path.exists(zipped_model_path)):
		zipdir(trained_model_dir_path, zipped_model_path)

	return send_file(zipped_model_path, attachment_filename=trained_model_file_name, as_attachment=True)
	


@app.route("/models/predict", methods=['POST'])
def Predict():
	# print("request.data:",request.data)

	raw_json = json.loads(request.data)
	
	dataset_json = json.loads(raw_json['selected_dataset_json'])

	input_image = PredictImagePreProcess(readb64(raw_json['input']))


	display_input_image = False

	if(display_input_image):
		# cv2_image = cv2.cvtColor(x[0], cv2.COLOR_RGB2BGR)
		cv2.imshow("image 0",input_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


	model_name = raw_json['selected_model']

	dataset_name = dataset_json["dataset_name"]

	# dataset_json = json.loads(request.form["selected_dataset_json"])

	# input_image = PredictImagePreProcess(readb64(request.form["input"]))
	# model_name = request.form["selected_model"]

	# dataset_name = dataset_json["dataset_name"]

	if(not model_name in loaded_models):
		loaded_models[model_name] = {}

	if(not dataset_name in loaded_models[model_name] ):
		loaded_models[model_name][dataset_name] = LoadModelFromName(model_name,dataset_json)

	prediction = loaded_models[model_name][dataset_name].Predict(np.array([input_image]))

	labels = [label["label"] for label in dataset_json["labels"]]
	labels.sort()
	print("raw prediction:", prediction)
	print("prediction:"+str(prediction[0])+" - "+labels[prediction[0]])
	json_data = json.dumps({'prediction': labels[prediction[0]]})

	return json_data



if __name__ == "__main__":
	print("load models jsons")

	#### load models json
	models_json_path = os.path.join(models_path,"models.json")

	models_json = None
	with open(models_json_path,"r") as f:
		models_json = json.load(f)

	loaded_models = {}
	
	print('Starting the API')
	app.run(host='0.0.0.0',port=6101)

