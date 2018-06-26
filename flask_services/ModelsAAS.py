from flask import Flask, Response ,send_file , send_from_directory
import os

import cv2

import base64
from PIL import Image
from StringIO import StringIO

import json

import sys


## Setup Sys path for easy imports
base_dir = "/media/harborned/ShutUpN/repos/dais/p5_afm_2018_demo"


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
	


# @app.route("/models/process_image", methods=['POST'])
# def ServeTestImages():
	
# 	response_dict = {"images":EncodeTestImages(dataset_name)}
# 	return json.dumps(response_dict)


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




if __name__ == "__main__":
	print("load models jsons")

	#### load models json
	models_json_path = os.path.join(models_path,"models.json")

	models_json = None
	with open(models_json_path,"r") as f:
		models_json = json.load(f)
	
	print('Starting the API')
	app.run(host='0.0.0.0',port=6101)

