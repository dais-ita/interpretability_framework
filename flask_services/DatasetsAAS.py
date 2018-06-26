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

#add dataset folder to sys path to allow for easy import
datasets_path = os.path.join(base_dir,"datasets")
sys.path.append(datasets_path)
###


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




def EncodeTestImages(dataset_name,num_images=1):
	pass


@app.route("/datasets/test_image/<string:dataset_name>", methods=['GET'])
def ServeTestImages(dataset_name):
	
	response_dict = {"images":EncodeTestImages(dataset_name)}
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

