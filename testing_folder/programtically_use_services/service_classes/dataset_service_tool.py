import os

import base64
from PIL import Image
from StringIO import StringIO

import cv2 
import numpy as np

import urllib2,urllib

import json


class DatasetServiceTool(object):
	"""docstring for DatasetServiceTool"""
	def __init__(self, service_config):
		super(DatasetServiceTool, self).__init__()
		

		self.service_json = [service for service in service_config["services"] if service["service_type"] == "dataset"][0]
		self.base_url = "http://"+self.service_json["ip"]+":"+self.service_json["port"]


	def GetAvailableDatasets(self):
		get_datasets_suffix = "/datasets/get_available"
		get_datasets_url = self.base_url + get_datasets_suffix
		return json.loads(urllib2.urlopen(get_datasets_url).read())


	def GetTestImage(self,dataset_name):
		request_suffix = "/datasets/test_image/{0}".format(dataset_name.replace(" ","%20"))
		request_url = self.base_url + request_suffix
		try:
			response = urllib2.urlopen(request_url)
		except  urllib2.URLError as e:
			print(request_url)
			print("request failed - error code:",e.code)
			return None
		json_response = json.loads(response.read())


		return json_response
