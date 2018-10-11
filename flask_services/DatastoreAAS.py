from flask import Flask, Response ,send_file , send_from_directory,request
from flask_cors import CORS, cross_origin

import os

import cv2

import base64
from PIL import Image
from StringIO import StringIO

import json

import sys

import numpy as np

import zipfile

import shutil

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
	


@app.route("/save_explanation_table", methods=['POST'])
@cross_origin()
def SaveExplanationTable():

	table_dict = {"explanation_table_data":{"results":[],"table_html":{"html":request.get_data()}}}

	file_path = "table_html.json"
	with open(file_path,"w") as f:
		f.write(json.dumps(table_dict))

	return "Saved as: "+file_path
	

@app.route("/save_explanation_table_from_json", methods=['POST'])
@cross_origin()
def SaveExplanationTableFromJson():

	file_path = "table_html.json"
	with open(file_path,"w") as f:
		f.write(request.get_data())

	return "Saved as: "+file_path


@app.route("/create_table_folder/<table_name>", methods=['GET'])
@cross_origin()
def CreateExplanationTableFolder(table_name):

	base_dir = "experiment_tables"
	
	table_folder_path = os.path.join(base_dir,table_name)

	if(not os.path.isdir(table_folder_path)):
		os.mkdir(table_folder_path)
	
	return table_folder_path


@app.route("/save_explanation_result_json", methods=['POST'])
@cross_origin()
def SaveExplanationResultJson():
	request.get_data()
	print("request.data",request.data)
	
	raw_json = json.loads(request.data)
	
	base_dir = raw_json["experiment_path"]
	explanation =  json.loads(raw_json["explanation"])
	json_file_name = explanation["dataset_identifier"]+"_"+explanation["image_identifier"]+"_"+explanation["model_identifier"]+"_"+explanation["explanation_identifier"]+".json"

	file_path = os.path.join(base_dir,json_file_name)

	with open(file_path,"w") as f:
		f.write(json.dumps(explanation))

	return file_path


@app.route("/compile_explanation_table_json", methods=['POST'])
@cross_origin()
def CompileExplanation():
	request.get_data()
	raw_json = json.loads(request.data)
	experiment_path = raw_json["experiment_path"]

	json_files = os.listdir(experiment_path)

	explanation_list = []
	explanation_lites = []
	for json_file in json_files:
		json_path = os.path.join(experiment_path,json_file)
		with open(json_path,"r") as f:
			explanation_json = json.load(f)
			explanation_list.append(explanation_json)
			explanation_lites.append(ExplanationJsonToJsonLite(explanation_json))

	table_json = {"explanation_table_scaffold":json.loads(raw_json["explanation_table_scaffold"]),"explanation_table_data":{"results":explanation_list}}
	table_json_lite = {"explanation_table_scaffold":json.loads(raw_json["explanation_table_scaffold"]),"explanation_table_data":{"results":explanation_lites}}

	output_path = os.path.join(experiment_path,"AAAexplanation_table.json")
	with open(output_path,"w") as f:
		f.write(json.dumps(table_json))

	output_path = os.path.join(experiment_path,"AAAexplanation_table_lite.json")
	with open(output_path,"w") as f:
		f.write(json.dumps(table_json_lite))

	return output_path



	
#LOADING TABLE
@app.route("/get_number_explanations_for_table/<table_name>", methods=['GET'])
@cross_origin()
def GetNumberOfExplanationsForTable(table_name):

	base_dir = "experiment_tables"
	
	experiment_path = os.path.join(base_dir,table_name)

	json_files = [f for f in os.listdir(experiment_path) if f[:3] != "AAA"]
	
	return str(len(json_files))


@app.route("/get_table_scaffold/<table_name>", methods=['GET'])
@cross_origin()
def GetTableScaffold(table_name):

	base_dir = "experiment_tables"
	
	table_folder_path = os.path.join(base_dir,table_name)

	json_file_path = os.path.join(table_folder_path,"AAAexplanation_table.json") 

	table_json_string = ""
	with open(json_file_path,"r") as f:
		table_json_string = f.read()

	table_json = json.loads(table_json_string)
	
	return json.dumps({"explanation_table_scaffold":table_json["explanation_table_scaffold"]})


@app.route("/get_table_explanation_by_index", methods=['POST'])
@cross_origin()
def GetTableExplanationByIndex():
	request.get_data()
	print("request.data",request.data)
	
	raw_json = json.loads(request.data)
	
	experiment_id = raw_json["experiment_id"]
	explanation_i =  raw_json["explanation_i"]

	base_dir = "experiment_tables"
	experiment_path = os.path.join(base_dir,experiment_id)

	json_files = [f for f in os.listdir(experiment_path) if f[:3] != "AAA"]

	file_path = os.path.join(experiment_path,json_files[int(explanation_i)])

	explanation_json = ""
	with open(file_path,"r") as f:
		explanation_json = f.read()

	return explanation_json

		

def ExplanationJsonToJsonLite(explanation_json):
	explanation_lite = {}
	explanation_lite["dataset_identifier"] = explanation_json["dataset_identifier"]
	explanation_lite["image_identifier"] = explanation_json["image_identifier"]
	explanation_lite["model_identifier"] = explanation_json["model_identifier"]
	explanation_lite["explanation_identifier"] = explanation_json["explanation_identifier"]
	explanation_lite["result_json"] = {"explanation_text":explanation_json["result_json"]["explanation_text"],"explanation_image":explanation_json["result_json"]["explanation_image"],"prediction":explanation_json["result_json"]["prediction"],"explanation_time":explanation_json["result_json"]["explanation_time"]}

	return explanation_lite


if __name__ == "__main__":
	print('Starting the API')
	app.run(host='0.0.0.0',port=6501)

