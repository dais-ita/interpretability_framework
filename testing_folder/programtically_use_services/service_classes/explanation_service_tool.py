import os

import base64
from PIL import Image
from StringIO import StringIO

import cv2 
import numpy as np

import urllib2,urllib

import json


class ExplanationServiceTool(object):
	"""docstring for ExplanationServiceTool"""
	def __init__(self, service_config):
		super(ExplanationServiceTool, self).__init__()
		

		self.service_json = [service for service in service_config["services"] if service["service_type"] == "explanation"][0]
		self.base_url = "http://"+self.service_json["ip"]+":"+self.service_json["port"]


	def GetExplanationsForModelAndDataset(self,dataset_name,model_name):
		filter_string = dataset_name+","+model_name
		request_suffix = "/explanations/get_available_for_filters/{0}".format(filter_string.replace(" ","%20"))
		request_url = self.base_url + request_suffix
		try:
			response = urllib2.urlopen(request_url)
		except  urllib2.URLError as e:
			print(request_url)
			if(code in e):
				print("request failed - error code:",e.code)
			else:
				print("request failed")
			return None
		json_response = json.loads(response.read())


		return json_response



	def ExplainPrediction(self,input_image,selected_dataset_json,selected_model_json,selected_explanation_json):
		request_suffix = "/explanations/explain"
		request_url = self.base_url + request_suffix

		values = {'input':input_image,'selected_dataset_json': selected_dataset_json,'selected_model_json': selected_model_json,'selected_explanation_json': selected_explanation_json}
		data = json.dumps(values)
		
		try:
			request = urllib2.Request(url=request_url,data=data)
			request.add_header("Content-Type","text/plain")
			response = urllib2.urlopen(request)
		except  urllib2.URLError as e:
			print(request_url)
			print("request failed - error code:",e.code)
			return None
		json_response = json.loads(response.read())


		return json_response