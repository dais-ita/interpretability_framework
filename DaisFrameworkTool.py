import sys
import os
import json

import cv2

import numpy as np
import random

import time



dataset_names = [
					"Gun Wielding Image Classification"
					,"Traffic Congestion Image Classification"
					,"Traffic Congestion Image Classification (Resized)"
					,"CIFAR-10"
					,"MNIST"
					,"ImageNet 20 Class (Resized)"
					,"ImageNet 10 Class (Resized)"
					,"ImageNet Vehicles Birds 10 Class (Resized)"
				]


class DaisFrameworkTool():
	def __init__(self,base_dir_name="interpretability_framework", explicit_framework_base_path=""):
		self.base_dir_name = base_dir_name
		
		if(explicit_framework_base_path==""):
			self.base_dir = self.GetProjectExplicitBase()
		else:
			self.base_dir = explicit_framework_base_path
		
		self.models_path = os.path.join(self.base_dir,"models")
		self.datasets_path = os.path.join(self.base_dir,"datasets")
		self.explanations_path = os.path.join(self.base_dir,"explanations")

		self.datasets_json = None
		self.models_json = None
		self.explanations_json = None


		self.model_save_path = None


		self.InitialiseTool()


	### Directory Handling Functions
	def GetProjectExplicitBase(self):
		cwd = os.getcwd()
		split_cwd = cwd.split("/")

		base_path_list = []
		for i in range(1, len(split_cwd)):
			if(split_cwd[-i] == self.base_dir_name):
				base_path_list = split_cwd[:-i+1]

		if(base_path_list == []):
			raise IOError('base project path could not be constructed. Are you running within: '+self.base_dir_name)

		base_dir_path = "/".join(base_path_list)

		return base_dir_path


	def AddDatasetsPathToSys(self):
		#add dataset folder to sys path to allow for easy import
		sys.path.append(self.datasets_path)

		#import dataset tool
		global DataSet
		from DatasetClass import DataSet



	def AddModelPathsToSys(self):
		#add all model folders to sys path to allow for easy import
		model_folders = os.listdir(self.models_path)

		for model_folder in model_folders:
			model_path = os.path.join(self.models_path,model_folder)
			sys.path.append(model_path)


	def AddExplanationPathsToSys(self):
		#add all explanation folders to sys path to allow for easy import
		explanation_folders = os.listdir(self.explanations_path)

		for explanation_folder in explanation_folders:
			explanation_path = os.path.join(self.explanations_path,explanation_folder)
			sys.path.append(explanation_path)


	###JSON Load Functions
	def LoadDatasetsJson(self):
		#### load dataset json
		data_json_path = os.path.join(self.datasets_path,"datasets.json")

		with open(data_json_path,"r") as f:
			self.datasets_json = json.load(f)


	def LoadModelsJson(self):
		### load model json
		model_json_path = os.path.join(self.models_path,"models.json")

		with open(model_json_path,"r") as f:
			self.models_json = json.load(f)


	def LoadExplanationsJson(self):
		### load model json
		explanation_json_path = os.path.join(self.explanations_path,"explanations.json")

		with open(explanation_json_path,"r") as f:
			self.explanations_json = json.load(f)


	def InitialiseTool(self):
		self.AddDatasetsPathToSys()
		self.AddModelPathsToSys()
		self.AddExplanationPathsToSys()


		self.LoadDatasetsJson()
		self.LoadModelsJson()
		self.LoadExplanationsJson()


	#DATASET FUNCTIONS
	def LoadFrameworkDataset(self,dataset_name):
		dataset_json = [dataset for dataset in self.datasets_json["datasets"] if dataset["dataset_name"] == dataset_name][0]

		### gather required information about the dataset
		if("default_training_allocation_path" in dataset_json.keys()):
			file_path = dataset_json["default_training_allocation_path"]
			load_split = True
		else:
			file_path = dataset_json["ground_truth_csv_path"]
			load_split = False
			print("new training split will be created")

		mean = None
		std = None
		
		if("mean" in dataset_json):
			mean = dataset_json["mean"]
		
		if("std" in dataset_json):
			std = dataset_json["std"]
		
		image_url_column = "image_path"
		ground_truth_column = "label"
		
		### instantiate dataset tool
		csv_path = os.path.join(self.datasets_path,"dataset_csvs",file_path)
		dataset_images_dir_path =  os.path.join(self.datasets_path,"dataset_images")

		dataset_tool = DataSet(csv_path,image_url_column,ground_truth_column,explicit_path_suffix=dataset_images_dir_path,mean=mean,std=std) #instantiates a dataset tool

		if(load_split):
			dataset_tool.ProduceDataFromTrainingSplitFile(csv_path,explicit_path_suffix =dataset_images_dir_path)
		else:
			dataset_tool.SplitLiveData(train_ratio=0.8,validation_ratio=0.1,test_ratio=0.1) #splits the live dataset examples in to train, validation and test sets
			dataset_tool.OutputTrainingSplitAllocation(csv_path.replace(".csv","_split.csv"))


		return dataset_json, dataset_tool

	#MODEL FUNCTIONS
	def InstantiateModelFromName(self,model_name,model_save_path_suffix,dataset_json,additional_args = {}):
		### instantiate the model
		print("instantiating model")
		
		model_json = [model for model in self.models_json["models"] if model["model_name"] == model_name ][0]
		print("selecting first model:" + model_json["model_name"])

		print(model_json["script_name"]+"."+model_json["class_name"])
		ModelModule = __import__(model_json["script_name"]) 
		ModelClass = getattr(ModelModule, model_json["class_name"])

		self.model_save_path = os.path.join(self.models_path,model_json["model_name"],"saved_models",dataset_json["dataset_name"].lower().replace(" ","_")+"_"+model_save_path_suffix)

		model_instance = ModelClass(dataset_json["image_y"], dataset_json["image_x"], dataset_json["image_channels"], len(dataset_json["labels"]), model_dir=self.model_save_path, additional_args=additional_args)

		return model_instance



	def TrainModel(self,model_instance,train_x, train_y, batch_size, num_train_steps, val_x= None, val_y=None):
		print("train model")
		print("")
		model_train_start_time = time.time()
		
		model_instance.TrainModel(train_x, train_y, batch_size, num_train_steps, val_x= val_x, val_y=val_y)
		model_train_time = time.time() - model_train_start_time
		print(model_train_time)

		if(self.model_save_path!=""):
			print("saving model")
			model_instance.SaveModel(self.model_save_path)
		
		accuracy_after_training = None
		if(not(val_x is None) and not(val_y is None)):
			accuracy_after_training = model_instance.EvaluateModel(val_x, val_y, batch_size)
			print(accuracy_after_training)

		
		completed_epochs = str(len(model_instance.model.history.history["loss"]))

		return {"training_time":model_train_time,"accuracy_after_training":accuracy_after_training,"completed_epochs":completed_epochs}

	#EXPLANTION FUNCTIONS
	def InstantiateExplanationFromName(self,explanation_name,model_instance):
		explanation_json = [explanation for explanation in self.explanations_json["explanations"] if explanation["explanation_name"] == explanation_name ][0]

		ExplanationModule = __import__(explanation_json["script_name"]) 
		ExplanationClass = getattr(ExplanationModule, explanation_json["class_name"])
		
		return ExplanationClass(model_instance)



if __name__ == '__main__':#
	#ARGUMENTS
	model_name = "vgg16_imagenet"
	model_save_path_suffix = "test_001" 

	dataset_name = "ImageNet Vehicles Birds 10 Class (Resized)"


	#TRAINING PARAMETERS
	learning_rate = 0.001
	batch_size = 64
	num_train_steps = 2


	#INSTANTIATE TOOL
	framework_tool = DaisFrameworkTool()


	#LOAD DATASET
	dataset_json, dataset_tool = framework_tool.LoadFrameworkDataset(dataset_name)

	label_names = [label["label"] for label in dataset_json["labels"]] # gets all labels in dataset. To use a subset of labels, build a list manually
	
	#LOAD TRAINING & VALIDATION DATA
	#load all train images as model handles batching
	print("load training data")
	print("")
	source = "train"
	train_x, train_y = dataset_tool.GetBatch(batch_size = -1,even_examples=True, y_labels_to_use=label_names, split_batch = True, split_one_hot = True, batch_source = source)
	
	print("num train examples: "+str(len(train_x)))


	#validate on 128 images only
	source = "validation"
	val_x, val_y = dataset_tool.GetBatch(batch_size = 256,even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source)
	print("num validation examples: "+str(len(val_x)))

	
	#INSTANTIATE MODEL
	model_instance = framework_tool.InstantiateModelFromName(model_name,model_save_path_suffix,dataset_json,additional_args = {"learning_rate":learning_rate})
	


	#TRAIN MODEL
	framework_tool.TrainModel(model_instance,train_x, train_y, batch_size, num_train_steps, val_x= val_x, val_y=val_y)