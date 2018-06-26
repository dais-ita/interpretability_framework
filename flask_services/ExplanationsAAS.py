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


#add all explanation folders to sys path to allow for easy import
explanations_path = os.path.join(base_dir,"explanations")

explanation_folders = os.listdir(explanations_path)

for explanation_folder in explanation_folders:
	explanation_path = os.path.join(explanations_path,explanation_folder)
	sys.path.append(explanation_path)



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




if __name__ == "__main__":
	print("load explanations jsons")

	#### load explanations json
	explanations_json_path = os.path.join(explanations_path,"explanations.json")

	explanations_json = None
	with open(explanations_json_path,"r") as f:
		explanations_json = json.load(f)
	
	print('Starting the API')
	app.run(host='0.0.0.0',port=6201)

